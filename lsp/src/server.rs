//! LSP server main loop using lsp-server crate (stdio transport).
//!
//! Supports: diagnostics, hover, goto-definition, completions,
//! semantic tokens, document symbols, and signature help.

use crate::analysis;
use crate::context::LspContext;
use crate::diagnostics::to_lsp_diagnostics;
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, GotoDefinition, HoverRequest, SemanticTokensFullRequest,
    SignatureHelpRequest,
};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionItemTag, CompletionOptions, CompletionParams,
    CompletionResponse, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, Location, MarkupContent,
    MarkupKind, OneOf, Position, PublishDiagnosticsParams, Range, SemanticToken,
    SemanticTokenModifier, SemanticTokenType, SemanticTokens, SemanticTokensFullOptions,
    SemanticTokensLegend, SemanticTokensOptions, SemanticTokensParams, SemanticTokensResult,
    SemanticTokensServerCapabilities, ServerCapabilities, SignatureHelp, SignatureHelpOptions,
    SignatureHelpParams, SignatureInformation, ParameterInformation, ParameterLabel,
    TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
};
use std::path::PathBuf;

/// Run the LSP server on stdio. Blocks until the client disconnects.
pub fn run_lsp() -> Result<(), String> {
    let (connection, io_threads) = Connection::stdio();

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
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::FULL,
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string(), ":".to_string()]),
            ..Default::default()
        }),
        document_symbol_provider: Some(OneOf::Left(true)),
        semantic_tokens_provider: Some(
            SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                legend: SemanticTokensLegend {
                    token_types,
                    token_modifiers,
                },
                full: Some(SemanticTokensFullOptions::Bool(true)),
                range: None,
                work_done_progress_options: Default::default(),
            }),
        ),
        signature_help_provider: Some(SignatureHelpOptions {
            trigger_characters: Some(vec!["(".to_string(), ",".to_string()]),
            retrigger_characters: Some(vec![",".to_string()]),
            work_done_progress_options: Default::default(),
        }),
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

    let root = init_params
        .root_uri
        .as_ref()
        .and_then(|uri| uri_to_file_path(uri.as_str()))
        .or_else(|| init_params.root_path.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let mut ctx = LspContext::new(root);
    ctx.load_workspace_config();

    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req).map_err(|e| e.to_string())? {
                    break;
                }
                handle_request(&connection, &ctx, req);
            }
            Message::Notification(not) => {
                handle_notification(&connection, &mut ctx, not);
            }
            Message::Response(_) => {}
        }
    }

    io_threads
        .join()
        .map_err(|e| format!("IO thread error: {}", e))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Request dispatch
// ---------------------------------------------------------------------------

fn handle_request(conn: &Connection, ctx: &LspContext, req: Request) {
    if let Some((id, params)) = cast_request::<HoverRequest>(&req) {
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

fn handle_notification(conn: &Connection, ctx: &mut LspContext, not: Notification) {
    if let Some(params) = cast_notification::<DidOpenTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        let text = params.text_document.text;
        ctx.open_files.insert(uri.clone(), text.clone());
        publish_diagnostics(conn, ctx, &uri, &text);
    } else if let Some(params) = cast_notification::<DidChangeTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        if let Some(change) = params.content_changes.into_iter().last() {
            ctx.open_files.insert(uri.clone(), change.text.clone());
            publish_diagnostics(conn, ctx, &uri, &change.text);
        }
    } else if let Some(params) = cast_notification::<DidCloseTextDocument>(&not) {
        let uri = params.text_document.uri.as_str().to_string();
        ctx.open_files.remove(&uri);
        let params = PublishDiagnosticsParams {
            uri: params.text_document.uri,
            diagnostics: vec![],
            version: None,
        };
        send_notification::<PublishDiagnostics>(conn, params);
    }
}

fn publish_diagnostics(conn: &Connection, ctx: &mut LspContext, uri: &str, source: &str) {
    let diags = ctx.compile_file(uri, source);
    let lsp_diags = to_lsp_diagnostics(&diags);

    let lsp_uri: Uri = uri
        .parse()
        .unwrap_or_else(|_| format!("file://{}", uri).parse().unwrap_or_else(|_| "file:///unknown".parse().unwrap()));

    let params = PublishDiagnosticsParams {
        uri: lsp_uri,
        diagnostics: lsp_diags,
        version: None,
    };
    send_notification::<PublishDiagnostics>(conn, params);
}

fn send_notification<N: lsp_types::notification::Notification>(conn: &Connection, params: N::Params)
where
    N::Params: serde::Serialize,
{
    let not = lsp_server::Notification::new(N::METHOD.to_string(), params);
    let _ = conn.sender.send(Message::Notification(not));
}

// ---------------------------------------------------------------------------
// Hover
// ---------------------------------------------------------------------------

fn handle_hover(ctx: &LspContext, params: HoverParams) -> Option<Hover> {
    let uri = params.text_document_position_params.text_document.uri.as_str();
    let pos = params.text_document_position_params.position;
    let info = ctx.hover_info(uri, pos.line + 1, pos.character + 1)?;

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
    let uri = params.text_document_position_params.text_document.uri.as_str();
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
// Completions — keywords + symbols
// ---------------------------------------------------------------------------

fn handle_completion(ctx: &LspContext, params: CompletionParams) -> Option<CompletionResponse> {
    let uri = params.text_document_position.text_document.uri.as_str();

    // Keywords from the parser
    let mut items: Vec<CompletionItem> = parser::HAXE_KEYWORDS
        .iter()
        .map(|kw| CompletionItem {
            label: kw.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            sort_text: Some(format!("9_{}", kw)),
            ..Default::default()
        })
        .collect();

    // Symbols from last compilation
    for entry in ctx.completions(uri) {
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

        items.push(item);
    }

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

    let symbols: Vec<DocumentSymbol> = entries
        .into_iter()
        .map(|e| doc_entry_to_lsp(e))
        .collect();

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

fn handle_signature_help(
    ctx: &LspContext,
    params: SignatureHelpParams,
) -> Option<SignatureHelp> {
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
