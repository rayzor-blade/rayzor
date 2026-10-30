//! Document tasks retain their stacks while other pages make progress.

use crate::context::LspContext;
use compiler::macro_system::{budget, display};
use krio_fiber::{Fiber, FiberStep};
use lsp_server::{Connection, Message};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

const SLICE: Duration = Duration::from_millis(8);
const STACK: usize = 256 << 20;

pub(crate) struct Task<R = LspContext> {
    pub uri: String,
    pub version: Option<i32>,
    pub fingerprint: Option<u64>,
    fiber: Fiber,
    budget: budget::State,
    display: display::State,
    result: Rc<RefCell<Option<R>>>,
    messages: Connection,
    progress: crossbeam_channel::Sender<Message>,
}

/// Unwinds a cancelled stack, releasing its compiler graph and guards.
struct Cancelled;

impl Task {
    pub fn new(mut context: LspContext, uri: String, allow_index: bool) -> Self {
        let version = context.document_versions.get(&uri).copied();
        let fingerprint = context.editor_cache.fingerprint(&uri, &context.open_files);
        let (messages, connection) = Connection::memory();
        let progress = connection.sender.clone();
        let result = Rc::new(RefCell::new(None));
        let output = result.clone();
        let document = uri.clone();
        let fiber = Fiber::with_stack_size(STACK, move || {
            if !krio_fiber::is_cancelled() {
                crate::server::analyze_document(&connection, &mut context, &document, allow_index);
            }
            *output.borrow_mut() = Some(context);
        });
        Self {
            uri,
            version,
            fingerprint,
            fiber,
            budget: Default::default(),
            display: Default::default(),
            result,
            messages,
            progress,
        }
    }
}

/// The completions a macro asks for at a cursor in inline markup.
pub(crate) type Display = Option<Vec<crate::analysis::CompletionEntry>>;

impl Task<Display> {
    /// Compiles `uri` with its display position at `cursor`, a character
    /// offset, leaving `context`'s analyses alone.
    pub fn display(context: LspContext, uri: String, cursor: u32) -> Self {
        let version = context.document_versions.get(&uri).copied();
        let (messages, connection) = Connection::memory();
        let progress = connection.sender.clone();
        let result = Rc::new(RefCell::new(None));
        let output = result.clone();
        let document = uri.clone();
        let fiber = Fiber::with_stack_size(STACK, move || {
            if krio_fiber::is_cancelled() {
                return;
            }
            let _budget = budget::limit(context.macro_budget);
            let source = context.open_files.get(&document).cloned();
            let answer = source.and_then(|source| {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    context.display_completions(&document, &source, cursor)
                }))
                .ok()
                .flatten()
            });
            *output.borrow_mut() = Some(answer);
        });
        Self {
            uri,
            version,
            fingerprint: None,
            fiber,
            budget: Default::default(),
            display: Default::default(),
            result,
            messages,
            progress,
        }
    }
}

impl<R> Task<R> {
    pub fn step(&mut self) -> FiberStep {
        let sender = self.progress.clone();
        let uri = self.uri.clone();
        let version = self.version;
        let _progress = compiler::cooperative::observe(Rc::new(move |activity| {
            let _ = sender.send(Message::Notification(lsp_server::Notification::new(
                "rayzor/indexProgress".into(),
                serde_json::json!({"uri":uri,"version":version,"phase":activity.phase,"subject":activity.subject,"completed":activity.completed,"total":activity.total}),
            )));
        }));
        let started = Rc::new(Cell::new(Instant::now()));
        let clock = started.clone();
        let _hook = compiler::cooperative::install(Rc::new(move || {
            if krio_fiber::is_cancelled() {
                std::panic::resume_unwind(Box::new(Cancelled));
            }
            if clock.get().elapsed() >= SLICE {
                krio_fiber::yield_now();
                clock.set(Instant::now());
                if krio_fiber::is_cancelled() {
                    std::panic::resume_unwind(Box::new(Cancelled));
                }
            }
        }));
        let host = budget::replace_state(std::mem::take(&mut self.budget));
        let shown = display::replace_state(std::mem::take(&mut self.display));
        let step = self.fiber.resume();
        self.display = display::replace_state(shown);
        self.budget = budget::replace_state(host);
        step
    }

    pub fn messages(&self) -> impl Iterator<Item = Message> + '_ {
        self.messages.receiver.try_iter()
    }

    pub fn take_result(&self) -> Option<R> {
        self.result.borrow_mut().take()
    }

    pub fn error(&mut self) -> Option<Box<dyn std::any::Any + Send>> {
        self.fiber.take_error()
    }

    pub fn cancel(&mut self) {
        self.fiber.cancel();
        // Every suspended task is stopped inside our checkpoint hook. Resume
        // it to unwind instead of freeing a stack containing live Rust values.
        while matches!(self.step(), FiberStep::Yielded) {}
    }
}

impl<R> Drop for Task<R> {
    fn drop(&mut self) {
        if matches!(
            self.fiber.state(),
            krio_fiber::FiberState::New | krio_fiber::FiberState::Suspended
        ) {
            self.cancel();
        }
    }
}
