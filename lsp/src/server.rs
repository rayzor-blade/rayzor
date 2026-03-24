//! LSP server main loop using lsp-server crate (stdio transport).

use crate::context::LspContext;
use crate::diagnostics::to_lsp_diagnostics;
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, PublishDiagnostics,
};
use lsp_types::request::{Completion, GotoDefinition, HoverRequest};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    HoverProviderCapability, InitializeParams, InitializeResult, Location, MarkupContent,
    MarkupKind, OneOf, Position, PublishDiagnosticsParams, Range, ServerCapabilities,
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

    let capabilities = ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(
            TextDocumentSyncKind::FULL,
        )),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        completion_provider: Some(CompletionOptions {
            trigger_characters: Some(vec![".".to_string()]),
            ..Default::default()
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

    // Determine workspace root from init params
    let root = init_params
        .root_uri
        .as_ref()
        .and_then(|uri| uri_to_file_path(uri.as_str()))
        .or_else(|| init_params.root_path.as_ref().map(PathBuf::from))
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

    let mut ctx = LspContext::new(root);
    ctx.load_workspace_config();

    // Main message loop
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

fn handle_request(conn: &Connection, ctx: &LspContext, req: Request) {
    if let Some((id, params)) = cast_request::<HoverRequest>(&req) {
        let result = handle_hover(ctx, params);
        let resp = Response::new_ok(id, result);
        let _ = conn.sender.send(Message::Response(resp));
    } else if let Some((id, params)) = cast_request::<GotoDefinition>(&req) {
        let result = handle_goto_definition(ctx, params);
        let resp = Response::new_ok(id, result);
        let _ = conn.sender.send(Message::Response(resp));
    } else if let Some((id, params)) = cast_request::<Completion>(&req) {
        let result = handle_completion(ctx, params);
        let resp = Response::new_ok(id, result);
        let _ = conn.sender.send(Message::Response(resp));
    } else {
        let resp = Response::new_ok(req.id, serde_json::Value::Null);
        let _ = conn.sender.send(Message::Response(resp));
    }
}

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
        // Clear diagnostics on close
        let params = PublishDiagnosticsParams {
            uri: params.text_document.uri,
            diagnostics: vec![],
            version: None,
        };
        let not = lsp_server::Notification::new(
            <PublishDiagnostics as lsp_types::notification::Notification>::METHOD.to_string(),
            params,
        );
        let _ = conn.sender.send(Message::Notification(not));
    }
}

fn publish_diagnostics(conn: &Connection, ctx: &mut LspContext, uri: &str, source: &str) {
    let diags = ctx.compile_file(uri, source);
    let lsp_diags = to_lsp_diagnostics(&diags);

    let lsp_uri: Uri = uri.parse().unwrap_or_else(|_| {
        format!("file://{}", uri)
            .parse()
            .unwrap_or_else(|_| "file:///unknown".parse().unwrap())
    });

    let params = PublishDiagnosticsParams {
        uri: lsp_uri,
        diagnostics: lsp_diags,
        version: None,
    };
    let not = lsp_server::Notification::new(
        <PublishDiagnostics as lsp_types::notification::Notification>::METHOD.to_string(),
        params,
    );
    let _ = conn.sender.send(Message::Notification(not));
}

fn handle_hover(ctx: &LspContext, params: HoverParams) -> Option<Hover> {
    let uri = params
        .text_document_position_params
        .text_document
        .uri
        .as_str();
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

fn handle_completion(ctx: &LspContext, params: CompletionParams) -> Option<CompletionResponse> {
    let uri = params
        .text_document_position
        .text_document
        .uri
        .as_str();

    // Haxe keywords
    let keywords = vec![
        "var", "function", "class", "interface", "enum", "abstract", "typedef", "import", "using",
        "if", "else", "for", "while", "do", "switch", "case", "default", "return", "break",
        "continue", "throw", "try", "catch", "new", "this", "super", "null", "true", "false",
        "public", "private", "static", "inline", "override", "dynamic", "extern",
    ];

    let mut items: Vec<CompletionItem> = keywords
        .into_iter()
        .map(|kw| CompletionItem {
            label: kw.to_string(),
            kind: Some(CompletionItemKind::KEYWORD),
            sort_text: Some(format!("2_{}", kw)), // Keywords sort after symbols
            ..Default::default()
        })
        .collect();

    // Symbol completions from the last compilation
    for entry in ctx.completions(uri) {
        items.push(CompletionItem {
            label: entry.label.clone(),
            kind: Some(entry.kind.to_lsp()),
            detail: Some(entry.detail),
            documentation: entry.documentation.map(|d| {
                lsp_types::Documentation::MarkupContent(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: d,
                })
            }),
            sort_text: Some(format!("1_{}", entry.label)), // Symbols sort before keywords
            ..Default::default()
        });
    }

    Some(CompletionResponse::Array(items))
}

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
