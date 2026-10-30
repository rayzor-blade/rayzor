//! LSP server main loop using lsp-server crate (stdio transport).
//!
//! Answers diagnostics, hover, definition, references, document highlight,
//! completion, semantic tokens, document symbols, signature help and inlay
//! hints; tests/session.rs checks which of them answer correctly.

use crate::analysis;
use crate::context::{Completions, LspContext};
use crate::diagnostics::to_lsp_diagnostics;
use lsp_server::{Connection, ErrorCode, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentHighlightRequest, DocumentSymbolRequest, GotoDefinition, HoverRequest,
    InlayHintRequest, References, SemanticTokensFullRequest, SignatureHelpRequest,
};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionItemTag, CompletionOptions, CompletionParams,
    CompletionResponse, DocumentHighlight, DocumentHighlightKind, DocumentHighlightParams,
    DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse, GotoDefinitionParams,
    GotoDefinitionResponse, Hover, HoverContents, HoverParams, HoverProviderCapability,
    InitializeParams, InitializeResult, InlayHint, InlayHintKind, InlayHintLabel, InlayHintParams,
    Location, MarkupContent, MarkupKind, OneOf, ParameterInformation, ParameterLabel, Position,
    PublishDiagnosticsParams, Range, ReferenceParams, SemanticToken, SemanticTokenModifier,
    SemanticTokenType, SemanticTokens, SemanticTokensFullOptions, SemanticTokensLegend,
    SemanticTokensOptions, SemanticTokensParams, SemanticTokensResult,
    SemanticTokensServerCapabilities, ServerCapabilities, SignatureHelp, SignatureHelpOptions,
    SignatureHelpParams, SignatureInformation, TextDocumentSyncCapability, TextDocumentSyncKind,
    Uri,
};
use std::path::PathBuf;

/// Deep recursion in the compiler cannot be caught, so the session runs on a
/// stack large enough for any program the command line compiles.
const SESSION_STACK: usize = 256 << 20;

/// Run the LSP server on stdio. Blocks until the client disconnects.
pub fn run_lsp() -> Result<(), String> {
    let (connection, io_threads) = crate::transport::stdio();
    let result = spawn_session(connection)
        .map_err(|e| format!("Cannot start the session: {e}"))?
        .join()
        .unwrap_or_else(|_| Err("the session panicked".to_string()));
    // The writer ends when the session drops its sender; the reader when the
    // client sends `exit` or closes stdin.
    for thread in io_threads {
        let _ = thread.join();
    }
    result
}

/// Serve one session on `connection` on its own thread, with the stack the
/// compiler needs.
pub fn spawn_session(
    connection: Connection,
) -> std::io::Result<std::thread::JoinHandle<Result<(), String>>> {
    std::thread::Builder::new()
        .name("lsp-session".into())
        .stack_size(SESSION_STACK)
        .spawn(move || serve(&connection))
}

/// Serve one session on `connection`, from `initialize` to `shutdown`.
pub fn serve(connection: &Connection) -> Result<(), String> {
    let cache = crate::editor_cache::EditorCache::default();
    let (front, worker_connection) = Connection::memory();
    let worker_cache = cache.clone();
    let worker = std::thread::Builder::new()
        .name("lsp-typing".into())
        .stack_size(SESSION_STACK)
        .spawn(move || serve_worker(&worker_connection, worker_cache))
        .map_err(|error| error.to_string())?;
    loop {
        crossbeam_channel::select! {
            recv(connection.receiver) -> message => {
                let Ok(message) = message else { break };
                if let Message::Notification(notification) = &message { cache.track(notification); }
                if let Message::Request(request) = &message
                    && let Some(result) = cache.answer(request)
                {
                    let _ = connection.sender.send(Message::Response(Response::new_ok(request.id.clone(), result)));
                    continue;
                }
                let exit = matches!(&message, Message::Notification(notification) if notification.method == "exit");
                if front.sender.send(message).is_err() || exit { break; }
            }
            recv(front.receiver) -> message => {
                let Ok(message) = message else { break };
                if connection.sender.send(message).is_err() { break; }
            }
        }
    }
    drop(front);
    worker
        .join()
        .unwrap_or_else(|_| Err("the typing session panicked".into()))
}

fn serve_worker(
    connection: &Connection,
    editor_cache: crate::editor_cache::EditorCache,
) -> Result<(), String> {
    let (id, params) = connection
        .initialize_start()
        .map_err(|e| format!("Initialize failed: {}", e))?;

    let init_params: InitializeParams =
        serde_json::from_value(params).map_err(|e| format!("Bad init params: {}", e))?;

    // Build semantic token legend from analysis constants
    let token_types: Vec<SemanticTokenType> = analysis::SEMANTIC_TOKEN_TYPES
        .iter()
        .map(|t| SemanticTokenType::new(t))
        .collect();
    let token_modifiers: Vec<SemanticTokenModifier> = analysis::SEMANTIC_TOKEN_MODIFIERS
        .iter()
        .map(|m| SemanticTokenModifier::new(m))
        .collect();

    let capabilities = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            // `<`, `/` and a space complete in markup only.
            trigger_characters: Some([".", ":", "<", "/", " "].map(String::from).to_vec()),
            ..Default::default()
        }),
        document_symbol_provider: Some(OneOf::Left(true)),
        semantic_tokens_provider: Some(SemanticTokensServerCapabilities::SemanticTokensOptions(
            SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types,
                    token_modifiers,
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: None,
                work_done_progress_options: Default::default(),
            },
        )),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: Some(vec![",".to_string()]),
            work_done_progress_options: Default::default(),
        }),
        inlay_hint_provider: Some(OneOf::Left(true)),
        experimental: Some(
            serde_json::json!({"macroExpansionProvider":true,"analysisStatusProvider":true,"diagnosticProgress":true,"partialIndexProvider":true}),
        ),
        references_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
        workspace: Some(lsp_types::WorkspaceServerCapabilities {
            workspace_folders: Some(lsp_types::WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: Some(OneOf::Left(true)),
            }),
            ..Default::default()
        }),
        document_highlight_provider: Some(OneOf::Left(true)),
        ..Default::default()
    };

    let init_result = InitializeResult {
        capabilities,
        server_info: Some(lsp_types::ServerInfo {
            name: "rayzor-lsp".to_string(),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        }),
    };

    let init_result_json =
        serde_json::to_value(init_result).map_err(|e| format!("Serialize error: {}", e))?;

    connection
        .initialize_finish(id, init_result_json)
        .map_err(|e| format!("Initialize finish failed: {}", e))?;

    #[allow(deprecated)]
    let root = init_params
        .root_uri
        .as_ref()
        .and_then(|uri| uri_to_file_path(uri.as_str()))
        .or_else(|| init_params.root_path.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let mut ctx = LspContext::new(root);
    ctx.editor_cache = editor_cache;
    if init_params
        .capabilities
        .experimental
        .as_ref()
        .and_then(|value| value.get("rayzorIndexProgress"))
        .and_then(|value| value.as_bool())
        == Some(true)
    {
        let sender = connection.sender.clone();
        ctx.index_progress = Some(std::sync::Arc::new(move |activity| {
            use crate::workspace_index::IndexActivity;
            let params = match activity {
                IndexActivity::Scanning {
                    path,
                    subject,
                    completed,
                } => {
                    serde_json::json!({"scope":"workspace","state":"checking","phase":"scanning","uri":crate::context::file_uri(&path.to_string_lossy()),"subject":subject,"completed":completed})
                }
                IndexActivity::Finished { completed } => {
                    serde_json::json!({"scope":"workspace","state":"ready","completed":completed})
                }
            };
            let _ = sender.send(Message::Notification(Notification::new(
                "rayzor/indexProgress".into(),
                params,
            )));
        }));
    }
    let hxml = init_params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("hxml"))
        .and_then(|h| h.as_str());
    if let Some(folders) = &init_params.workspace_folders {
        ctx.workspace_roots = folders
            .iter()
            .filter_map(|folder| uri_to_file_path(folder.uri.as_str()))
            .collect();
    }
    ctx.load_workspace_config(hxml);
    ctx.analysis_cache_capacity = init_params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("analysisCacheSize"))
        .and_then(|size| size.as_u64())
        .unwrap_or(2)
        .clamp(1, 8) as usize;
    ctx.macro_budget = init_params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("macroBudgetMs"))
        .and_then(|ms| ms.as_u64())
        .filter(|ms| *ms != 0)
        .map(std::time::Duration::from_millis);
    ctx.configure_editor_cache();
    publish_configuration_warnings(connection, &mut ctx);
    for problem in &ctx.config_problems {
        log_message(connection, problem);
    }
    if init_params
        .capabilities
        .workspace
        .as_ref()
        .and_then(|w| w.did_change_watched_files.as_ref())
        .and_then(|w| w.dynamic_registration)
        == Some(true)
    {
        let mut watchers: Vec<_> = [
            "**/*.hx",
            "**/*.hxml",
            "**/rayzor.toml",
            "**/haxelib.json",
            "**/*.rpkg",
        ]
        .into_iter()
        .map(|pattern| serde_json::json!({"globPattern":pattern}))
        .collect();
        watchers.extend(ctx.configuration_watch_paths().into_iter().map(
            |path| serde_json::json!({"globPattern":path.to_string_lossy().replace('\\', "/")}),
        ));
        let request = Request::new(
            "rayzor-file-watch".to_string().into(),
            "client/registerCapability".into(),
            serde_json::json!({"registrations":[{"id":"rayzor-haxe-files","method":"workspace/didChangeWatchedFiles","registerOptions":{"watchers":watchers}}]}),
        );
        let _ = connection.sender.send(Message::Request(request));
    }

    let mut pending = std::collections::VecDeque::new();
    let mut tasks = std::collections::VecDeque::<crate::indexing::Task>::new();
    // Display compiles for completion in markup, with the cursor each asks at.
    let mut displays =
        std::collections::VecDeque::<(u32, crate::indexing::Task<crate::indexing::Display>)>::new();
    let mut deferred = Vec::new();
    loop {
        ctx.poll_workspace_index();
        let first = if tasks.is_empty() && pending.is_empty() && displays.is_empty() {
            match connection
                .receiver
                .recv_timeout(std::time::Duration::from_millis(100))
            {
                Ok(message) => Some(message),
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => None,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        } else {
            match connection.receiver.try_recv() {
                Ok(message) => Some(message),
                Err(crossbeam_channel::TryRecvError::Empty) => None,
                Err(crossbeam_channel::TryRecvError::Disconnected) => break,
            }
        };
        let mut changed = Vec::new();
        let mut requests = Vec::new();
        for message in first.into_iter().chain(connection.receiver.try_iter()) {
            match message {
                Message::Request(request) => requests.push(request),
                Message::Notification(notification) if notification.method == "$/cancelRequest" => {
                    let id = &notification.params["id"];
                    for queue in [&mut requests, &mut deferred] {
                        queue.retain(|request: &Request| {
                            if serde_json::to_value(&request.id).ok().as_ref() == Some(id) {
                                let _ =
                                    connection.sender.send(Message::Response(Response::new_err(
                                        request.id.clone(),
                                        -32800,
                                        "Request cancelled".into(),
                                    )));
                                false
                            } else {
                                true
                            }
                        });
                    }
                }
                Message::Notification(notification) => {
                    let invalidate = match notification.method.as_str() {
                        "textDocument/didChange"
                        | "workspace/didChangeWatchedFiles"
                        | "workspace/didChangeConfiguration"
                        | "workspace/didChangeWorkspaceFolders" => true,
                        "textDocument/didOpen" => {
                            let uri = notification.params["textDocument"]["uri"].as_str();
                            let text = notification.params["textDocument"]["text"].as_str();
                            uri.zip(text).is_some_and(|(uri, text)| {
                                std::fs::read_to_string(crate::context::uri_to_path(uri))
                                    .ok()
                                    .as_deref()
                                    != Some(text)
                            })
                        }
                        "textDocument/didClose" => notification.params["textDocument"]["uri"]
                            .as_str()
                            .and_then(|uri| ctx.open_files.get(uri).map(|text| (uri, text)))
                            .is_some_and(|(uri, text)| {
                                std::fs::read_to_string(crate::context::uri_to_path(uri))
                                    .ok()
                                    .as_ref()
                                    != Some(text)
                            }),
                        "textDocument/didSave" => notification.params["textDocument"]["uri"]
                            .as_str()
                            .is_some_and(|uri| !uri.ends_with(".hx")),
                        _ => false,
                    };
                    handle_notification(connection, &mut ctx, notification, &mut changed);
                    if invalidate {
                        displays.clear();
                        let mut interrupted = Vec::new();
                        for mut task in tasks.drain(..) {
                            task.cancel();
                            if ctx.open_files.contains_key(&task.uri) {
                                interrupted.push(task.uri.clone());
                            }
                        }
                        interrupted.append(&mut changed);
                        changed = interrupted;
                    }
                }
                Message::Response(_) => {}
            }
        }
        pending.retain(|(uri, _)| ctx.open_files.contains_key(uri));
        let mut seen = std::collections::HashSet::new();
        let changed: Vec<_> = changed
            .into_iter()
            .rev()
            .filter(|uri| seen.insert(uri.clone()))
            .collect();
        for uri in changed.into_iter().rev() {
            if let Some(snapshot) = ctx
                .editor_cache
                .lookup(&uri)
                .filter(|snapshot| snapshot.complete)
            {
                publish_diagnostic_updates(
                    connection,
                    ctx.update_diagnostics(&uri, snapshot.diagnostics.clone()),
                    ctx.document_versions
                        .get(&uri)
                        .copied()
                        .map(|version| (uri.as_str(), version)),
                );
                analysis_status(
                    connection,
                    &uri,
                    ctx.document_versions.get(&uri).copied(),
                    "ready",
                );
                continue;
            }
            enqueue_document(connection, &ctx, &mut pending, &mut tasks, uri, true);
        }
        for request in requests {
            if request.method == "shutdown" {
                for task in &mut tasks {
                    task.cancel();
                }
                answer_request(connection, &ctx, request)?;
                return Ok(());
            }
            let uri = request.params["textDocument"]["uri"]
                .as_str()
                .map(str::to_string);
            if let Some(uri) = uri
                && ctx.open_files.contains_key(&uri)
                && ctx.needs_analysis(&uri)
                && ctx.editor_cache.answer(&request).is_none()
            {
                enqueue_document(connection, &ctx, &mut pending, &mut tasks, uri, false);
            }
            deferred.push(request);
        }
        while tasks.len() < 2 {
            let Some((uri, allow_index)) = pending.pop_front() else {
                break;
            };
            if ctx.open_files.contains_key(&uri) {
                let context = ctx.fork_analysis();
                tasks.push_back(crate::indexing::Task::new(context, uri, allow_index));
            }
        }
        if let Some(mut task) = tasks.pop_front() {
            if !ctx.open_files.contains_key(&task.uri)
                || ctx.document_versions.get(&task.uri).copied() != task.version
            {
                task.cancel();
            } else {
                let step = task.step();
                for message in task.messages() {
                    if let Message::Notification(note) = &message
                        && (note.method == "rayzor/indexProgress"
                            || note.method == "window/logMessage"
                            || note.method == "rayzor/analysisStatus"
                                && note.params["state"] == "checking")
                    {
                        let _ = connection.sender.send(message);
                    }
                }
                if step == krio_fiber::FiberStep::Yielded {
                    tasks.push_back(task);
                } else if task.fingerprint.is_some()
                    && task.fingerprint != ctx.editor_cache.fingerprint(&task.uri, &ctx.open_files)
                {
                    pending.push_back((task.uri.clone(), true));
                } else {
                    let diagnostics = if let Some(context) = task.take_result() {
                        ctx.adopt_analysis(&task.uri, context)
                    } else {
                        ctx.mark_analysis_failed(&task.uri);
                        let message = task
                            .error()
                            .map(|error| panic_message(&error))
                            .unwrap_or_else(|| "indexing task failed".into());
                        vec![crate::context::LspDiagnostic::error(
                            &crate::context::uri_to_path(&task.uri),
                            1,
                            1,
                            message,
                        )]
                    };
                    publish_diagnostic_updates(
                        connection,
                        ctx.update_diagnostics(&task.uri, diagnostics),
                        task.version.map(|version| (task.uri.as_str(), version)),
                    );
                    analysis_status(connection, &task.uri, task.version, "ready");
                }
            }
        }
        if let Some((cursor, mut task)) = displays.pop_front()
            && ctx.document_versions.get(&task.uri).copied() == task.version
        {
            if task.step() == krio_fiber::FiberStep::Yielded {
                displays.push_back((cursor, task));
            } else {
                let entries = task.take_result().flatten();
                ctx.store_display_answer(&task.uri, task.version, cursor, entries);
            }
        }
        let mut waiting = Vec::new();
        for request in deferred.drain(..) {
            let display = display_request(&ctx, &request);
            if display.is_none() && markup_trigger(&request) {
                send_response(connection, request.id, Some(Vec::<CompletionItem>::new()));
            } else if let Some((uri, cursor)) = display.clone()
                && !ctx.has_display_answer(&uri, cursor)
            {
                if !displays
                    .iter()
                    .any(|(asked, task)| *asked == cursor && task.uri == uri)
                {
                    let task = crate::indexing::Task::display(ctx.fork_settings(), uri, cursor);
                    displays.push_back((cursor, task));
                }
                waiting.push(request);
            } else if display
                .is_some_and(|(uri, cursor)| ctx.display_entries(&uri, cursor).is_some())
            {
                if answer_request(connection, &ctx, request)? {
                    return Ok(());
                }
            } else if let Some(result) = ctx.editor_cache.answer(&request) {
                send_response(connection, request.id, Some(result));
            } else if request.params["textDocument"]["uri"]
                .as_str()
                .is_some_and(|uri| {
                    tasks.iter().any(|task| task.uri == uri)
                        || pending.iter().any(|(document, _)| document == uri)
                })
            {
                waiting.push(request);
            } else if answer_request(connection, &ctx, request)? {
                return Ok(());
            }
        }
        deferred = waiting;
    }
    Ok(())
}

/// Whether a completion request was triggered by a character that completes
/// in markup only.
pub(crate) fn markup_trigger(request: &Request) -> bool {
    request.method == "textDocument/completion"
        && matches!(
            request.params["context"]["triggerCharacter"].as_str(),
            Some("<" | "/" | " ")
        )
}

/// The document and markup cursor of a completion request a display
/// compile answers.
fn display_request(ctx: &LspContext, request: &Request) -> Option<(String, u32)> {
    if request.method != "textDocument/completion" {
        return None;
    }
    let uri = request.params["textDocument"]["uri"].as_str()?;
    let line = request.params["position"]["line"].as_u64()? as u32;
    let column = request.params["position"]["character"].as_u64()? as u32;
    ctx.display_cursor(uri, line + 1, column + 1)
        .map(|cursor| (uri.to_string(), cursor))
}

fn enqueue_document(
    connection: &Connection,
    ctx: &LspContext,
    pending: &mut std::collections::VecDeque<(String, bool)>,
    tasks: &mut std::collections::VecDeque<crate::indexing::Task>,
    uri: String,
    allow_index: bool,
) {
    if let Some(index) = tasks.iter().position(|task| task.uri == uri) {
        if tasks[index].version == ctx.document_versions.get(&uri).copied() {
            return;
        }
        let mut task = tasks.remove(index).unwrap();
        task.cancel();
    }
    let was_queued = pending.iter().any(|(document, _)| *document == uri);
    let allow_index = if let Some(index) = pending.iter().position(|(document, _)| *document == uri)
    {
        let previous = pending.remove(index).unwrap().1;
        allow_index && previous
    } else {
        allow_index
    };
    // Reserve a slot for newly active work; unfinished background pages remain queued.
    if tasks.len() >= 2 {
        let mut background = tasks.pop_back().unwrap();
        background.cancel();
        pending.push_back((background.uri.clone(), true));
    }
    if !was_queued {
        analysis_status(
            connection,
            &uri,
            ctx.document_versions.get(&uri).copied(),
            "queued",
        );
    }
    pending.push_front((uri, allow_index));
}

fn answer_request(connection: &Connection, ctx: &LspContext, req: Request) -> Result<bool, String> {
    if connection
        .handle_shutdown(&req)
        .map_err(|e| e.to_string())?
    {
        return Ok(true);
    }
    let id = req.id.clone();
    let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        handle_request(connection, ctx, req)
    }));
    if let Err(payload) = answered {
        let message = format!("internal error: {}", panic_message(&payload));
        log_message(connection, &message);
        let failed = Response::new_err(id, ErrorCode::InternalError as i32, message);
        let _ = connection.sender.send(Message::Response(failed));
    }
    Ok(false)
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "a panic".to_string())
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

fn handle_request(conn: &Connection, ctx: &LspContext, req: Request) {
    if req.method == "workspace/symbol" {
        let query = req
            .params
            .get("query")
            .and_then(|q| q.as_str())
            .unwrap_or_default();
        send_response(conn, req.id, Some(ctx.workspace_symbols(query)));
    } else if req.method == "rayzor/expandMacro" {
        let preview = serde_json::from_value::<lsp_types::TextDocumentPositionParams>(req.params)
            .ok()
            .and_then(|p| ctx.expand_macro(p.text_document.uri.as_str(), p.position));
        send_response(conn, req.id, preview);
    } else if let Some((id, params)) = cast_request::<HoverRequest>(&req) {
        send_response(conn, id, handle_hover(ctx, params));
    } else if let Some((id, params)) = cast_request::<GotoDefinition>(&req) {
        send_response(conn, id, handle_goto_definition(ctx, params));
    } else if let Some((id, params)) = cast_request::<Completion>(&req) {
        send_response(conn, id, handle_completion(ctx, params));
    } else if let Some((id, params)) = cast_request::<SemanticTokensFullRequest>(&req) {
        send_response(conn, id, handle_semantic_tokens(ctx, params));
    } else if let Some((id, params)) = cast_request::<DocumentSymbolRequest>(&req) {
        send_response(conn, id, handle_document_symbols(ctx, params));
    } else if let Some((id, params)) = cast_request::<SignatureHelpRequest>(&req) {
        send_response(conn, id, handle_signature_help(ctx, params));
    } else if let Some((id, params)) = cast_request::<InlayHintRequest>(&req) {
        send_response(conn, id, handle_inlay_hints(ctx, params));
    } else if let Some((id, params)) = cast_request::<References>(&req) {
        send_response(conn, id, handle_references(ctx, params));
    } else if let Some((id, params)) = cast_request::<DocumentHighlightRequest>(&req) {
        send_response(conn, id, handle_document_highlight(ctx, params));
    } else {
        send_response::<serde_json::Value>(conn, req.id, None);
    }
}

fn send_response<T: serde::Serialize>(conn: &Connection, id: RequestId, result: Option<T>) {
    let resp = match result {
        Some(val) => Response::new_ok(id, val),
        None => Response::new_ok(id, serde_json::Value::Null),
    };
    let _ = conn.sender.send(Message::Response(resp));
}

// ---------------------------------------------------------------------------
// Notifications
// ---------------------------------------------------------------------------

/// Applies a notification; a document whose text changed is added to
/// `changed` once, to be compiled after the batch.
fn handle_notification(
    conn: &Connection,
    ctx: &mut LspContext,
    not: Notification,
    changed: &mut Vec<String>,
) {
    let mut mark = |uri: &str| {
        if !changed.iter().any(|u| u == uri) {
            changed.push(uri.to_string());
        }
    };
    if let Some(params) = cast_notification::<DidOpenTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        ctx.document_versions
            .insert(uri.clone(), params.text_document.version);
        ctx.open_files
            .insert(uri.clone(), params.text_document.text);
        mark(&uri);
    } else if let Some(params) = cast_notification::<DidChangeTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        if let Some(change) = params.content_changes.into_iter().last() {
            ctx.document_versions
                .insert(uri.clone(), params.text_document.version);
            ctx.open_files.insert(uri.clone(), change.text);
            mark(&uri);
        }
    } else if let Some(params) = cast_notification::<DidCloseTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        publish_diagnostic_updates(conn, ctx.update_diagnostics(&uri, Vec::new()), None);
        ctx.close(&uri);
        changed.retain(|u| *u != uri);
        let params = PublishDiagnosticsParams {
            uri: params.text_document.uri,
            diagnostics: vec![],
            version: None,
        };
        send_notification::<PublishDiagnostics>(conn, params);
    } else if let Some(params) =
        cast_notification::<lsp_types::notification::DidChangeWorkspaceFolders>(&not)
    {
        for folder in params.event.removed {
            if let Some(path) = uri_to_file_path(folder.uri.as_str()) {
                let path = path.canonicalize().unwrap_or(path);
                ctx.workspace_roots
                    .retain(|root| root.canonicalize().unwrap_or_else(|_| root.clone()) != path);
            }
        }
        for folder in params.event.added {
            if let Some(path) = uri_to_file_path(folder.uri.as_str()) {
                let canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
                if !ctx
                    .workspace_roots
                    .iter()
                    .any(|root| root.canonicalize().unwrap_or_else(|_| root.clone()) == canonical)
                {
                    ctx.workspace_roots.push(path);
                }
            }
        }
        ctx.reload_workspace_config();
        publish_configuration_warnings(conn, ctx);
        for uri in ctx.open_files.keys() {
            mark(uri);
        }
    } else if cast_notification::<lsp_types::notification::DidChangeConfiguration>(&not).is_some() {
        ctx.reload_workspace_config();
        publish_configuration_warnings(conn, ctx);
        for uri in ctx.open_files.keys() {
            mark(uri);
        }
        for problem in &ctx.config_problems {
            log_message(conn, problem);
        }
    } else if let Some(params) =
        cast_notification::<lsp_types::notification::DidSaveTextDocument>(&not)
    {
        let uri = params.text_document.uri.as_str();
        if !params.text_document.uri.path().as_str().ends_with(".hx") {
            ctx.reload_workspace_config();
            publish_configuration_warnings(conn, ctx);
            for uri in ctx.open_files.keys() {
                mark(uri);
            }
        } else {
            if let Some(text) = params.text {
                ctx.open_files.insert(uri.into(), text);
            }
            ctx.index_disk_file(uri);
            if !ctx.has_current_analysis(uri) {
                mark(uri);
            }
        }
    } else if let Some(params) =
        cast_notification::<lsp_types::notification::DidChangeWatchedFiles>(&not)
    {
        if params
            .changes
            .iter()
            .any(|change| !change.uri.path().as_str().ends_with(".hx"))
        {
            ctx.reload_workspace_config();
            publish_configuration_warnings(conn, ctx);
            for uri in ctx.open_files.keys() {
                mark(uri);
            }
            for problem in &ctx.config_problems {
                log_message(conn, problem);
            }
        } else {
            for change in params.changes {
                ctx.index_disk_file(change.uri.as_str());
            }
            ctx.invalidate_changed_inputs();
            for uri in ctx.open_files.keys() {
                if !ctx.has_current_analysis(uri) && ctx.editor_cache.lookup(uri).is_none() {
                    mark(uri);
                }
            }
        }
    }
}

pub(crate) fn analyze_document(
    conn: &Connection,
    ctx: &mut LspContext,
    uri: &str,
    allow_index: bool,
) {
    let Some(source) = ctx.open_files.get(uri).cloned() else {
        return;
    };
    let version = ctx.document_versions.get(uri).copied();
    if allow_index
        && !ctx.has_current_analysis(uri)
        && let Some(snapshot) = ctx
            .editor_cache
            .lookup(uri)
            .filter(|snapshot| snapshot.complete)
    {
        publish_diagnostic_updates(
            conn,
            ctx.update_diagnostics(uri, snapshot.diagnostics.clone()),
            version.map(|version| (uri, version)),
        );
        analysis_status(conn, uri, version, "ready");
        return;
    }
    let input_hash = ctx.editor_cache.fingerprint(uri, &ctx.open_files);
    analysis_status(conn, uri, version, "checking");
    if let Some(input_hash) = input_hash
        && let Ok(snapshot) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            declaration_snapshot(ctx, uri, &source)
        }))
    {
        let (snapshot, subject) = snapshot;
        let count = snapshot.names.len();
        ctx.editor_cache.publish_partial(uri, input_hash, snapshot);
        let _ = conn.sender.send(Message::Notification(Notification::new("rayzor/indexProgress".into(), serde_json::json!({"uri":uri,"version":version,"phase":"declarations","subject":subject,"completed":count}))));
    }
    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _budget = compiler::macro_system::budget::limit(ctx.macro_budget);
        ctx.compile_file(uri, &source)
    }));
    let diags = compiled.unwrap_or_else(|payload| {
        ctx.invalidate_analysis_cache();
        let message = format!("internal compiler error: {}", panic_message(&payload));
        log_message(conn, &message);
        vec![crate::context::LspDiagnostic::error(
            &crate::context::uri_to_path(uri),
            1,
            1,
            message,
        )]
    });
    let cached_diagnostics = diags.clone();
    publish_diagnostic_updates(
        conn,
        ctx.update_diagnostics(uri, diags),
        version.map(|version| (uri, version)),
    );
    if ctx.has_current_analysis(uri)
        && let Some(input_hash) = input_hash
        && ctx.editor_cache.fingerprint(uri, &ctx.open_files) == Some(input_hash)
    {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build_editor_snapshot(conn, ctx, uri, input_hash, cached_diagnostics)
        })) {
            Ok(snapshot)
                if ctx.editor_cache.fingerprint(uri, &ctx.open_files) == Some(input_hash) =>
            {
                ctx.editor_cache.save(uri, input_hash, snapshot)
            }
            Ok(_) => {}
            Err(payload) => log_message(
                conn,
                &format!(
                    "Editor index failed: {}; serving from live analysis",
                    panic_message(&payload)
                ),
            ),
        }
    }
    analysis_status(conn, uri, version, "ready");
}

fn build_editor_snapshot(
    conn: &Connection,
    ctx: &LspContext,
    uri: &str,
    input_hash: u64,
    diagnostics: Vec<crate::context::LspDiagnostic>,
) -> crate::editor_cache::Snapshot {
    use serde_json::{Value, json};
    let (server, client) = Connection::memory();
    let render = |method: &str, params: Value| {
        handle_request(&server, ctx, Request::new(0.into(), method.into(), params));
        match client.receiver.recv() {
            Ok(Message::Response(response)) => response.result.unwrap_or(Value::Null),
            _ => Value::Null,
        }
    };
    let mut snapshot = crate::editor_cache::Snapshot {
        diagnostics,
        dependencies: ctx.macro_dependencies(uri),
        expansions: ctx
            .indexed_expansions(uri)
            .into_iter()
            .filter_map(crate::editor_cache::Expansion::from_json)
            .collect(),
        ..Default::default()
    };
    let declarations = ctx
        .open_files
        .get(uri)
        .map(|source| declaration_snapshot(ctx, uri, source))
        .unwrap_or_default();
    let (declarations, subject) = declarations;
    for method in [
        "textDocument/documentSymbol",
        "textDocument/semanticTokens/full",
        "textDocument/completion",
        "textDocument/inlayHint",
    ] {
        let params = json!({"textDocument":{"uri":uri},"position":{"line":0,"character":0},"range":{"start":{"line":0,"character":0},"end":{"line":u32::MAX,"character":0}}});
        snapshot.document.insert(method, render(method, params));
    }
    let names = ctx.indexed_names(uri);
    let total = names.len();
    let mut records = std::collections::HashMap::new();
    let mut member_records = std::collections::HashMap::new();
    for (line, column, len, symbol) in names {
        compiler::cooperative::checkpoint();
        let params =
            json!({"textDocument":{"uri":uri},"position":{"line":line,"character":column}});
        let metadata = if let Some(&record) = records.get(&symbol) {
            record
        } else {
            let mut metadata = crate::binary_response::Responses::default();
            for method in ["textDocument/definition", "textDocument/documentHighlight"] {
                metadata.insert(method, render(method, params.clone()));
            }
            for include in [false, true] {
                let mut params = params.clone();
                params["context"] = json!({"includeDeclaration":include});
                metadata.insert(
                    &format!("textDocument/references:{include}"),
                    render("textDocument/references", params),
                );
            }
            let record = snapshot.records.len();
            snapshot.records.push(metadata);
            records.insert(symbol, record);
            record
        };
        let responses = crate::binary_response::Responses::from([(
            "textDocument/hover",
            render("textDocument/hover", params),
        )]);
        let completion = json!({"textDocument":{"uri":uri},"position":{"line":line,"character":column + len + 1}});
        if ctx
            .open_files
            .get(uri)
            .and_then(|source| analysis::receiver_before(source, line + 1, column + len + 2))
            .is_some_and(|(row, col, _)| row == line + 1 && col == column + 1)
        {
            let record = if let Some(&record) = member_records.get(&symbol) {
                record
            } else {
                let record = snapshot.member_results.len();
                snapshot
                    .member_results
                    .push(render("textDocument/completion", completion).into());
                member_records.insert(symbol, record);
                record
            };
            snapshot.members.insert((line, column), record);
        }
        snapshot.names.push(crate::editor_cache::Name {
            line,
            column,
            len,
            metadata: Some(metadata),
            responses,
        });
        if snapshot.names.len().is_multiple_of(128) {
            let mut partial = snapshot.clone();
            for declaration in &declarations.names {
                if !partial
                    .names
                    .iter()
                    .any(|name| name.line == declaration.line && name.column == declaration.column)
                {
                    partial.names.push(declaration.clone());
                }
            }
            ctx.editor_cache.publish_partial(uri, input_hash, partial);
            let _ = conn.sender.send(Message::Notification(Notification::new("rayzor/indexProgress".into(), json!({"uri":uri,"version":ctx.document_versions.get(uri),"phase":"types","subject":subject,"completed":snapshot.names.len(),"total":total}))));
        }
    }
    if let Some(source) = ctx.open_files.get(uri)
        && let Ok(tokens) = parser::lexer::Lexer::new(source).tokenize()
    {
        for token in tokens
            .iter()
            .filter(|token| token.kind == parser::token::TokenKind::LParen)
        {
            let before = &source[..token.span.end];
            let line = before.bytes().filter(|&byte| byte == b'\n').count() as u32;
            let column = before
                .rsplit('\n')
                .next()
                .unwrap_or("")
                .encode_utf16()
                .count() as u32;
            if let Some(call) = analysis::enclosing_call(source, line + 1, column + 1)
                && let Some(name) = snapshot.names.iter_mut().find(|name| {
                    name.line + 1 == call.callee_line && name.column + 1 == call.callee_col
                })
            {
                let params =
                    json!({"textDocument":{"uri":uri},"position":{"line":line,"character":column}});
                name.responses.insert(
                    "textDocument/signatureHelp",
                    render("textDocument/signatureHelp", params),
                );
            }
        }
    }
    snapshot.dependencies = ctx.macro_dependencies(uri);
    snapshot.complete = true;
    snapshot
}

fn declaration_snapshot(
    ctx: &LspContext,
    uri: &str,
    source: &str,
) -> (crate::editor_cache::Snapshot, Option<String>) {
    use serde_json::json;
    let filename = crate::context::uri_to_path(uri);
    let settings = ctx.settings_for_uri(uri);
    let defines: Vec<_> = settings
        .defines
        .iter()
        .chain(&settings.options.extra_defines)
        .cloned()
        .collect();
    let declarations =
        crate::workspace_index::declarations(std::path::Path::new(&filename), source, &defines);
    let mut snapshot = crate::editor_cache::Snapshot::default();
    let subject = declarations
        .iter()
        .find(|entry| entry.type_kind.is_some())
        .map(|entry| entry.qualified.clone());
    let mut completions = Vec::new();
    let mut outline = Vec::new();
    for declaration in declarations {
        let mut hover = format!("```haxe\n{}\n```", declaration.signature);
        if let Some(doc) = &declaration.documentation {
            hover.push_str("\n\n");
            hover.push_str(doc);
        }
        let mut responses = crate::binary_response::Responses::default();
        responses.insert(
            "textDocument/hover",
            json!({"contents":{"kind":"markdown","value":hover},"range":declaration.range}),
        );
        responses.insert(
            "textDocument/definition",
            json!({"uri":uri,"range":declaration.range}),
        );
        responses.mark_syntax();
        let kind = match declaration.kind {
            lsp_types::SymbolKind::CLASS => CompletionItemKind::CLASS,
            lsp_types::SymbolKind::INTERFACE => CompletionItemKind::INTERFACE,
            lsp_types::SymbolKind::STRUCT => CompletionItemKind::STRUCT,
            lsp_types::SymbolKind::ENUM => CompletionItemKind::ENUM,
            lsp_types::SymbolKind::METHOD | lsp_types::SymbolKind::FUNCTION => {
                CompletionItemKind::FUNCTION
            }
            _ => CompletionItemKind::FIELD,
        };
        completions.push(json!({"label":declaration.name,"kind":kind,"detail":declaration.signature,"documentation":declaration.documentation.map(|doc| json!({"kind":"markdown","value":doc}))}));
        outline.push(json!({"name":declaration.name,"detail":declaration.signature,"kind":declaration.kind,"range":declaration.range,"selectionRange":declaration.range}));
        snapshot.names.push(crate::editor_cache::Name {
            line: declaration.range.start.line,
            column: declaration.range.start.character,
            len: declaration
                .range
                .end
                .character
                .saturating_sub(declaration.range.start.character),
            metadata: None,
            responses,
        });
    }
    snapshot.document.insert(
        "textDocument/completion",
        json!({"isIncomplete":true,"items":completions}),
    );
    snapshot
        .document
        .insert("textDocument/documentSymbol", json!(outline));
    snapshot.document.mark_syntax();
    (snapshot, subject)
}

fn analysis_status(conn: &Connection, uri: &str, version: Option<i32>, state: &str) {
    let notification = Notification::new(
        "rayzor/analysisStatus".into(),
        serde_json::json!({"uri":uri,"version":version,"state":state}),
    );
    let _ = conn.sender.send(Message::Notification(notification));
}

/// Tells the client about a configuration problem it can show in its log.
fn log_message(conn: &Connection, message: &str) {
    let params = lsp_types::LogMessageParams {
        typ: lsp_types::MessageType::WARNING,
        message: message.to_string(),
    };
    send_notification::<lsp_types::notification::LogMessage>(conn, params);
}

fn send_notification<N: lsp_types::notification::Notification>(conn: &Connection, params: N::Params)
where
    N::Params: serde::Serialize,
{
    let not = lsp_server::Notification::new(N::METHOD.to_string(), params);
    let _ = conn.sender.send(Message::Notification(not));
}

fn publish_configuration_warnings(conn: &Connection, ctx: &mut LspContext) {
    let warnings = ctx.configuration_warnings();
    let owner = crate::context::file_uri(&ctx.root.to_string_lossy());
    let updates = ctx.update_diagnostics(&owner, warnings);
    publish_diagnostic_updates(conn, updates, None);
}

fn publish_diagnostic_updates(
    conn: &Connection,
    updates: Vec<(String, Vec<crate::context::LspDiagnostic>)>,
    analyzed: Option<(&str, i32)>,
) {
    for (file, diagnostics) in updates {
        let Ok(uri) = crate::context::file_uri(&file).parse::<lsp_types::Uri>() else {
            continue;
        };
        send_notification::<PublishDiagnostics>(
            conn,
            PublishDiagnosticsParams {
                version: analyzed
                    .filter(|(owner, _)| *owner == uri.as_str())
                    .map(|(_, version)| version),
                uri,
                diagnostics: to_lsp_diagnostics(&diagnostics),
            },
        );
    }
}

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

fn handle_hover(ctx: &LspContext, params: HoverParams) -> Option<Hover> {
    let uri = params
        .text_document_position_params
        .text_document
        .uri
        .as_str();
    let pos = params.text_document_position_params.position;
    let info = ctx.hover_info(uri, pos.line + 1, pos.character + 1);
    if info.is_none()
        && let Some(preview) = ctx.expand_macro(uri, pos)
        && let Some(expansion) = preview["expansion"]
            .as_str()
            .filter(|text| !text.trim().is_empty())
    {
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "**Macro expansion: `{}`**\n\n```haxe\n{}\n```",
                    preview["name"].as_str().unwrap_or("macro"),
                    expansion
                ),
            }),
            range: serde_json::from_value(preview["range"].clone()).ok(),
        });
    }
    let info = info?;

    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: info,
        }),
        range: None,
    })
}

// ---------------------------------------------------------------------------
// Go-to-definition
// ---------------------------------------------------------------------------

fn handle_goto_definition(
    ctx: &LspContext,
    params: GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let uri = params
        .text_document_position_params
        .text_document
        .uri
        .as_str();
    let pos = params.text_document_position_params.position;
    let (file, line, col) = ctx.goto_definition(uri, pos.line + 1, pos.character + 1)?;

    let target_uri: Uri = crate::context::file_uri(&file).parse().ok()?;
    let target_pos = Position::new(line.saturating_sub(1), col.saturating_sub(1));

    Some(GotoDefinitionResponse::Scalar(Location {
        uri: target_uri,
        range: Range::new(target_pos, target_pos),
    }))
}

// ---------------------------------------------------------------------------
// References and highlights
// ---------------------------------------------------------------------------

fn name_range(line: u32, column: u32, len: u32) -> Range {
    let start = Position::new(line.saturating_sub(1), column.saturating_sub(1));
    Range::new(start, Position::new(start.line, start.character + len))
}

fn handle_references(ctx: &LspContext, params: ReferenceParams) -> Option<Vec<Location>> {
    let doc = &params.text_document_position;
    let pos = doc.position;
    let found = ctx.references(
        doc.text_document.uri.as_str(),
        pos.line + 1,
        pos.character + 1,
        params.context.include_declaration,
    );
    let locations: Vec<Location> = found
        .into_iter()
        .filter_map(|(file, line, column, len)| {
            Some(Location {
                uri: format!("file://{}", file).parse().ok()?,
                range: name_range(line, column, len),
            })
        })
        .collect();
    (!locations.is_empty()).then_some(locations)
}

fn handle_document_highlight(
    ctx: &LspContext,
    params: DocumentHighlightParams,
) -> Option<Vec<DocumentHighlight>> {
    let doc = &params.text_document_position_params;
    let uri = doc.text_document.uri.as_str();
    let path = uri_to_file_path(uri)?;
    let pos = doc.position;
    let highlights: Vec<DocumentHighlight> = ctx
        .references(uri, pos.line + 1, pos.character + 1, true)
        .into_iter()
        .filter(|(file, ..)| std::path::Path::new(file) == path)
        .map(|(_, line, column, len)| DocumentHighlight {
            range: name_range(line, column, len),
            kind: Some(DocumentHighlightKind::TEXT),
        })
        .collect();
    (!highlights.is_empty()).then_some(highlights)
}

// ---------------------------------------------------------------------------
// Completions — keywords + symbols
// ---------------------------------------------------------------------------

fn handle_completion(ctx: &LspContext, params: CompletionParams) -> Option<CompletionResponse> {
    let doc = &params.text_document_position;
    let uri = doc.text_document.uri.as_str();
    let pos = doc.position;
    let to_item = |entry: analysis::CompletionEntry| {
        let mut item = CompletionItem {
            label: entry.label.clone(),
            kind: Some(entry.kind.to_lsp()),
            detail: Some(entry.detail),
            documentation: entry.documentation.map(|d| {
                lsp_types::Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: d,
                })
            }),
            sort_text: Some(format!("{}_{}", entry.sort_priority, entry.label)),
            ..Default::default()
        };
        if entry.deprecated {
            item.tags = Some(vec![CompletionItemTag::DEPRECATED]);
        }
        item
    };
    let items = match ctx.completions(uri, pos.line + 1, pos.character + 1) {
        Completions::Members(members) => members.into_iter().map(to_item).collect(),
        Completions::Names(names) => parser::HAXE_KEYWORDS
            .iter()
            .map(|kw| CompletionItem {
                label: kw.to_string(),
                kind: Some(CompletionItemKind::KEYWORD),
                sort_text: Some(format!("9_{}", kw)),
                ..Default::default()
            })
            .chain(names.into_iter().map(to_item))
            .collect(),
    };
    Some(CompletionResponse::Array(items))
}

// ---------------------------------------------------------------------------
// Semantic tokens — full syntax highlighting
// ---------------------------------------------------------------------------

fn handle_semantic_tokens(
    ctx: &LspContext,
    params: SemanticTokensParams,
) -> Option<SemanticTokensResult> {
    let uri = params.text_document.uri.as_str();
    let tokens = ctx.semantic_tokens(uri)?;

    // Encode as LSP delta format: each token is relative to the previous one
    let mut encoded = Vec::new();
    let mut prev_line = 0u32;
    let mut prev_start = 0u32;

    for tok in &tokens {
        let delta_line = tok.line - prev_line;
        let delta_start = if delta_line == 0 {
            tok.start_char - prev_start
        } else {
            tok.start_char
        };

        encoded.push(SemanticToken {
            delta_line,
            delta_start,
            length: tok.length,
            token_type: tok.token_type,
            token_modifiers_bitset: tok.token_modifiers,
        });

        prev_line = tok.line;
        prev_start = tok.start_char;
    }

    Some(SemanticTokensResult::Tokens(SemanticTokens {
        result_id: None,
        data: encoded,
    }))
}

// ---------------------------------------------------------------------------
// Document symbols — outline view
// ---------------------------------------------------------------------------

fn handle_document_symbols(
    ctx: &LspContext,
    params: DocumentSymbolParams,
) -> Option<DocumentSymbolResponse> {
    let uri = params.text_document.uri.as_str();
    let entries = ctx.document_symbols(uri)?;

    let symbols: Vec<DocumentSymbol> = entries.into_iter().map(doc_entry_to_lsp).collect();

    Some(DocumentSymbolResponse::Nested(symbols))
}

#[allow(deprecated)] // DocumentSymbol::deprecated field
fn doc_entry_to_lsp(entry: analysis::DocumentSymbolEntry) -> DocumentSymbol {
    let range = Range::new(
        Position::new(entry.line.saturating_sub(1), entry.col.saturating_sub(1)),
        Position::new(
            entry.end_line.saturating_sub(1),
            entry.end_col.saturating_sub(1),
        ),
    );
    let children = if entry.children.is_empty() {
        None
    } else {
        Some(entry.children.into_iter().map(doc_entry_to_lsp).collect())
    };

    DocumentSymbol {
        name: entry.name,
        detail: Some(entry.detail),
        kind: entry.kind,
        tags: None,
        deprecated: None,
        range,
        selection_range: range,
        children,
    }
}

// ---------------------------------------------------------------------------
// Signature help — function parameter info
// ---------------------------------------------------------------------------

fn handle_signature_help(ctx: &LspContext, params: SignatureHelpParams) -> Option<SignatureHelp> {
    let uri = params
        .text_document_position_params
        .text_document
        .uri
        .as_str();
    let pos = params.text_document_position_params.position;

    let sig = ctx.signature_help(uri, pos.line + 1, pos.character + 1)?;

    let parameters: Vec<ParameterInformation> = sig
        .parameters
        .iter()
        .map(|p| ParameterInformation {
            label: ParameterLabel::Simple(p.label.clone()),
            documentation: p.documentation.as_ref().map(|d| {
                lsp_types::Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: d.clone(),
                })
            }),
        })
        .collect();

    Some(SignatureHelp {
        signatures: vec![SignatureInformation {
            label: sig.label,
            documentation: sig.documentation.map(|d| {
                lsp_types::Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: d,
                })
            }),
            parameters: Some(parameters),
            active_parameter: Some(sig.active_parameter),
        }],
        active_signature: Some(0),
        active_parameter: Some(sig.active_parameter),
    })
}

// ---------------------------------------------------------------------------
// Inlay hints — macro expansion results + type annotations
// ---------------------------------------------------------------------------

fn handle_inlay_hints(ctx: &LspContext, params: InlayHintParams) -> Option<Vec<InlayHint>> {
    let uri = params.text_document.uri.as_str();
    let range = params.range;

    let mut hints = Vec::new();

    // Macro expansion hints: show expanded result after macro call sites
    for hint in ctx.macro_expansion_hints(uri) {
        let line = hint.line.saturating_sub(1); // LSP is 0-based
        let col = hint.column.saturating_sub(1);

        // Filter to requested range
        if line < range.start.line || line > range.end.line {
            continue;
        }

        // Show as: `macroName(...)` → `expanded_result`
        let compact = hint
            .expanded_text
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let label = format!(
            " → {}{}",
            compact.chars().take(80).collect::<String>(),
            if compact.chars().count() > 80 {
                "…"
            } else {
                ""
            }
        );

        hints.push(InlayHint {
            position: Position::new(line, col),
            label: InlayHintLabel::String(label),
            kind: Some(InlayHintKind::PARAMETER),
            text_edits: None,
            tooltip: Some(lsp_types::InlayHintTooltip::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "**Macro expansion** `{}`\n\n```haxe\n{}\n```",
                    hint.macro_name, hint.expanded_text
                ),
            })),
            padding_left: Some(true),
            padding_right: None,
            data: None,
        });
    }

    if hints.is_empty() { None } else { Some(hints) }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn uri_to_file_path(uri: &str) -> Option<PathBuf> {
    uri.strip_prefix("file://")
        .map(|p| PathBuf::from(p.replace("%20", " ")))
}

fn cast_request<R>(req: &Request) -> Option<(RequestId, R::Params)>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    if req.method == R::METHOD {
        let params: R::Params = serde_json::from_value(req.params.clone()).ok()?;
        Some((req.id.clone(), params))
    } else {
        None
    }
}

fn cast_notification<N>(not: &Notification) -> Option<N::Params>
where
    N: lsp_types::notification::Notification,
    N::Params: serde::de::DeserializeOwned,
{
    if not.method == N::METHOD {
        serde_json::from_value(not.params.clone()).ok()
    } else {
        None
    }
}
