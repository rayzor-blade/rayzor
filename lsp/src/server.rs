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

/// How long all the macros of one compile may run before they are stopped,
/// unless the client sets `initializationOptions.macroBudgetMs`.
const MACRO_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

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
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
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
        experimental: Some(serde_json::json!({"macroExpansionProvider":true})),
        references_provider: Some(OneOf::Left(true)),
        workspace_symbol_provider: Some(OneOf::Left(true)),
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
    let hxml = init_params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("hxml"))
        .and_then(|h| h.as_str());
    ctx.load_workspace_config(hxml);
    ctx.macro_budget = init_params
        .initialization_options
        .as_ref()
        .and_then(|o| o.get("macroBudgetMs"))
        .and_then(|ms| ms.as_u64())
        .map_or(MACRO_BUDGET, std::time::Duration::from_millis);
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
        let request = Request::new(
            "rayzor-file-watch".to_string().into(),
            "client/registerCapability".into(),
            serde_json::json!({"registrations":[{"id":"rayzor-haxe-files","method":"workspace/didChangeWatchedFiles","registerOptions":{"watchers":[{"globPattern":"**/*.hx"}]}}]}),
        );
        let _ = connection.sender.send(Message::Request(request));
    }

    // Everything queued is taken at once: the edits first, each changed
    // document compiled once at its latest text, then the requests, which so
    // see that text.
    let mut changed: Vec<String> = Vec::new();
    while let Ok(first) = connection.receiver.recv() {
        let mut requests = Vec::new();
        for msg in std::iter::once(first).chain(connection.receiver.try_iter()) {
            match msg {
                Message::Request(req) => requests.push(req),
                Message::Notification(not) => {
                    handle_notification(connection, &mut ctx, not, &mut changed)
                }
                Message::Response(_) => {}
            }
        }
        for uri in changed.drain(..) {
            publish_diagnostics(connection, &mut ctx, &uri);
        }
        for req in requests {
            if connection
                .handle_shutdown(&req)
                .map_err(|e| e.to_string())?
            {
                return Ok(());
            }
            let id = req.id.clone();
            let answered = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                handle_request(connection, &ctx, req)
            }));
            if let Err(payload) = answered {
                let message = format!("internal error: {}", panic_message(&payload));
                log_message(connection, &message);
                let failed = Response::new_err(id, ErrorCode::InternalError as i32, message);
                let _ = connection.sender.send(Message::Response(failed));
            }
        }
    }
    Ok(())
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
        ctx.open_files
            .insert(uri.clone(), params.text_document.text);
        mark(&uri);
    } else if let Some(params) = cast_notification::<DidChangeTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        if let Some(change) = params.content_changes.into_iter().last() {
            ctx.open_files.insert(uri.clone(), change.text);
            mark(&uri);
        }
    } else if let Some(params) = cast_notification::<DidCloseTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        publish_diagnostic_updates(conn, ctx.update_diagnostics(&uri, Vec::new()));
        ctx.close(&uri);
        changed.retain(|u| *u != uri);
        let params = PublishDiagnosticsParams {
            uri: params.text_document.uri,
            diagnostics: vec![],
            version: None,
        };
        send_notification::<PublishDiagnostics>(conn, params);
    } else if let Some(params) =
        cast_notification::<lsp_types::notification::DidChangeWatchedFiles>(&not)
    {
        for change in params.changes {
            ctx.index_disk_file(change.uri.as_str());
        }
    }
}

/// Compiles `uri` at its current text and publishes what it finds. A panic
/// inside the compiler becomes a diagnostic; the last analysis stays.
fn publish_diagnostics(conn: &Connection, ctx: &mut LspContext, uri: &str) {
    let Some(source) = ctx.open_files.get(uri).cloned() else {
        return;
    };
    let compiled = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _budget = compiler::macro_system::budget::limit(Some(ctx.macro_budget));
        ctx.compile_file(uri, &source)
    }));
    let diags = compiled.unwrap_or_else(|payload| {
        let message = format!("internal compiler error: {}", panic_message(&payload));
        log_message(conn, &message);
        vec![crate::context::LspDiagnostic::error(
            &crate::context::uri_to_path(uri),
            1,
            1,
            message,
        )]
    });
    publish_diagnostic_updates(conn, ctx.update_diagnostics(uri, diags));
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

fn publish_diagnostic_updates(
    conn: &Connection,
    updates: Vec<(String, Vec<crate::context::LspDiagnostic>)>,
) {
    for (file, diagnostics) in updates {
        let Ok(uri) = crate::context::file_uri(&file).parse() else {
            continue;
        };
        send_notification::<PublishDiagnostics>(
            conn,
            PublishDiagnosticsParams {
                uri,
                diagnostics: to_lsp_diagnostics(&diagnostics),
                version: None,
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
        && let Some(expansion) = preview["expansion"].as_str()
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

    let target_uri: Uri = format!("file://{}", file).parse().ok()?;
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
