//! Document tasks retain their stacks while other pages make progress.

use crate::context::LspContext;
use compiler::macro_system::budget;
use krio_fiber::{Fiber, FiberStep};
use lsp_server::{Connection, Message};
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::time::{Duration, Instant};

const SLICE: Duration = Duration::from_millis(8);
const STACK: usize = 256 << 20;

pub(crate) struct Task {
    pub uri: String,
    pub version: Option<i32>,
    pub fingerprint: Option<u64>,
    fiber: Fiber,
    budget: budget::State,
    result: Rc<RefCell<Option<LspContext>>>,
    messages: Connection,
}

/// Unwinds a cancelled stack, releasing its compiler graph and guards.
struct Cancelled;

impl Task {
    pub fn new(mut context: LspContext, uri: String, allow_index: bool) -> Self {
        let version = context.document_versions.get(&uri).copied();
        let fingerprint = context.editor_cache.fingerprint(&uri, &context.open_files);
        let (messages, connection) = Connection::memory();
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
            result,
            messages,
        }
    }

    pub fn step(&mut self) -> FiberStep {
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
        let step = self.fiber.resume();
        self.budget = budget::replace_state(host);
        step
    }

    pub fn messages(&self) -> impl Iterator<Item = Message> + '_ {
        self.messages.receiver.try_iter()
    }

    pub fn take_result(&self) -> Option<LspContext> {
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

impl Drop for Task {
    fn drop(&mut self) {
        if matches!(
            self.fiber.state(),
            krio_fiber::FiberState::New | krio_fiber::FiberState::Suspended
        ) {
            self.cancel();
        }
    }
}
