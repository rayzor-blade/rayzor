//! Custom error type that captures context strings from nom's context() combinator

use nom::error::{ContextError, ErrorKind, FromExternalError, ParseError};

/// A context with its associated byte offset
#[derive(Debug, Clone, PartialEq)]
pub struct ContextWithLocation {
    pub context: &'static str,
    pub byte_offset: usize,
}

/// Custom error type that captures context strings
#[derive(Debug, Clone, PartialEq)]
pub struct ContextualError<I> {
    pub input: I,
    pub code: ErrorKind,
    /// Context strings with their locations
    pub contexts: Vec<ContextWithLocation>,
    /// The byte offset of the actual parse failure
    pub byte_offset: Option<usize>,
}

// Thread-local storage for the full input to calculate byte offsets
thread_local! {
    static FULL_INPUT: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
    static DEEPEST_ERROR: std::cell::RefCell<Option<(Vec<ContextWithLocation>, usize)>> = const { std::cell::RefCell::new(None) };
}

/// Set the full input for automatic byte offset calculation
pub fn set_full_input(input: &str) {
    FULL_INPUT.with(|i| {
        *i.borrow_mut() = Some(input.to_string());
    });
}

/// Clear the full input
pub fn clear_full_input() {
    FULL_INPUT.with(|i| {
        *i.borrow_mut() = None;
    });
    DEEPEST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

/// Get the deepest error if any
pub fn get_deepest_error() -> Option<(Vec<ContextWithLocation>, usize)> {
    DEEPEST_ERROR.with(|e| e.borrow().clone())
}

impl<I> ContextualError<I> {
    pub fn new(input: I, code: ErrorKind) -> Self {
        Self {
            input,
            code,
            contexts: Vec::new(),
            byte_offset: None,
        }
    }

    pub fn with_byte_offset(mut self, offset: usize) -> Self {
        self.byte_offset = Some(offset);
        self
    }
}

impl<I> ParseError<I> for ContextualError<I>
where
    I: AsRef<str>,
{
    fn from_error_kind(input: I, kind: ErrorKind) -> Self {
        // println!("from_error_kind: creating NEW error at pos (first 20): {:?}, kind: {:?}",
        //          &input.as_ref()[..20.min(input.as_ref().len())], kind);
        Self {
            input,
            code: kind,
            contexts: Vec::new(),
            byte_offset: None,
        }
    }

    fn or(self, other: Self) -> Self {
        // Called by alt() to combine errors from different branches
        // Keep the error with the deepest context (furthest parse progress)

        // println!("or: combining alt branch errors");
        // println!("    self has {} contexts", self.contexts.len());
        // if let Some(deepest) = self.contexts.iter().max_by_key(|c| c.byte_offset) {
        //     println!("    self deepest: offset {} - '{}'", deepest.byte_offset, deepest.context);
        // }
        // println!("    other has {} contexts", other.contexts.len());
        // if let Some(deepest) = other.contexts.iter().max_by_key(|c| c.byte_offset) {
        //     println!("    other deepest: offset {} - '{}'", deepest.byte_offset, deepest.context);
        // }

        // Find the deepest context in each error
        let self_deepest = self
            .contexts
            .iter()
            .map(|c| c.byte_offset)
            .max()
            .or(self.byte_offset)
            .unwrap_or(0);

        let other_deepest = other
            .contexts
            .iter()
            .map(|c| c.byte_offset)
            .max()
            .or(other.byte_offset)
            .unwrap_or(0);

        // Keep the error with deeper progress
        if self_deepest > other_deepest {
            // println!("    keeping self (offset {} > {})", self_deepest, other_deepest);
            self
        } else if other_deepest > self_deepest {
            // println!("    keeping other (offset {} > {})", other_deepest, self_deepest);
            other
        } else {
            // Same depth - prefer the one with more contexts
            if self.contexts.len() >= other.contexts.len() {
                // println!("    keeping self (same depth, {} contexts >= {})", self.contexts.len(), other.contexts.len());
                self
            } else {
                // println!("    keeping other (same depth, {} contexts > {})", other.contexts.len(), self.contexts.len());
                other
            }
        }
    }

    fn append(input: I, kind: ErrorKind, other: Self) -> Self {
        // When backtracking, nom calls this to combine errors
        // This is called by alt() when trying different branches
        // println!("append: input pos (first 20 chars): {:?}, kind: {:?}",
        //          &input.as_ref()[..20.min(input.as_ref().len())], kind);
        // println!("        other has {} contexts, byte_offset: {:?}",
        //          other.contexts.len(), other.byte_offset);
        //
        // // Debug: show deepest context in other
        // if let Some(deepest) = other.contexts.iter().max_by_key(|c| c.byte_offset) {
        //     println!("        deepest context in other: offset {} - '{}'",
        //              deepest.byte_offset, deepest.context);
        // }

        // Calculate byte offset for the new input position
        let new_byte_offset = FULL_INPUT.with(|full| {
            if let Some(full_str) = full.borrow().as_ref() {
                full_str.find(input.as_ref()).unwrap_or(0)
            } else {
                0
            }
        });

        // println!("        new byte_offset would be: {}", new_byte_offset);

        // Find the deepest context offset in the accumulated error
        let deepest_context_offset = other
            .contexts
            .iter()
            .map(|c| c.byte_offset)
            .max()
            .unwrap_or(0);

        // Keep the error with the deepest progress (either by error offset or context offset)
        let other_deepest = other.byte_offset.unwrap_or(0).max(deepest_context_offset);

        if other_deepest > new_byte_offset {
            // The accumulated error or its contexts are deeper, keep it
            // println!("        keeping other (deepest {} > {})", other_deepest, new_byte_offset);
            return other;
        } else if other_deepest == new_byte_offset {
            // Same depth, keep the one with contexts
            if !other.contexts.is_empty() {
                // println!("        keeping other (same offset, has contexts)");
                return other;
            }
        }

        // The new position is deeper, but preserve deep contexts from other
        let mut preserved_contexts = Vec::new();
        for ctx in other.contexts {
            if ctx.byte_offset >= new_byte_offset {
                // Keep contexts that are at least as deep as the new position
                preserved_contexts.push(ctx);
            }
        }

        // println!("        creating new error at offset {}, preserving {} deep contexts",
        //          new_byte_offset, preserved_contexts.len());
        Self {
            input,
            code: kind,
            contexts: preserved_contexts,
            byte_offset: Some(new_byte_offset),
        }
    }
}

impl<I> ContextError<I> for ContextualError<I>
where
    I: AsRef<str>,
{
    fn add_context(_input: I, ctx: &'static str, mut other: Self) -> Self {
        // This is called when a parser with context() fails
        // Calculate the byte offset for this context based on the error input position
        let byte_offset = FULL_INPUT.with(|full| {
            if let Some(full_str) = full.borrow().as_ref() {
                let error_input = other.input.as_ref();
                full_str.find(error_input).unwrap_or(0)
            } else {
                0
            }
        });

        // println!("add_context: '{}' at offset {}", ctx, byte_offset);

        // Add the context with its location
        other.contexts.push(ContextWithLocation {
            context: ctx,
            byte_offset,
        });

        // Set the error's byte offset if not already set
        if other.byte_offset.is_none() {
            other.byte_offset = Some(byte_offset);
        }

        // Check if this is the deepest error we've seen and store it
        DEEPEST_ERROR.with(|e| {
            let mut deepest = e.borrow_mut();
            let should_update = if let Some((_, current_deepest)) = deepest.as_ref() {
                byte_offset > *current_deepest
            } else {
                true
            };

            if should_update {
                // println!("    -> Recording as deepest error (offset {})", byte_offset);
                *deepest = Some((other.contexts.clone(), byte_offset));
            }
        });

        other
    }
}

impl<I, E> FromExternalError<I, E> for ContextualError<I> {
    fn from_external_error(input: I, kind: ErrorKind, _e: E) -> Self {
        Self::new(input, kind)
    }
}

impl<I: std::fmt::Display> std::fmt::Display for ContextualError<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // If we have context, show the deepest one
        if let Some(deepest) = self.contexts.iter().max_by_key(|c| c.byte_offset) {
            write!(f, "{}", deepest.context)?;
        } else {
            write!(f, "error {:?} at: {}", self.code, self.input)?;
        }
        Ok(())
    }
}

impl<I> ContextualError<I> {
    /// Convert this error to a diagnostic using the actual context error message
    pub fn to_diagnostic(&self, span: diagnostics::SourceSpan) -> diagnostics::Diagnostic {
        use crate::error_syntax::ParsedError;
        use diagnostics::DiagnosticBuilder;

        // First check if we have a deepest error stored
        let contexts_to_use = if let Some((deep_contexts, _offset)) = get_deepest_error() {
            // println!("Using deepest stored error at offset {}", offset);
            deep_contexts
        } else {
            // println!("No deepest error stored, using current error's contexts");
            self.contexts.clone()
        };

        // Find the context that has the deepest byte offset (the actual failure point)
        let context_str = contexts_to_use
            .iter()
            .max_by_key(|c| c.byte_offset)
            .map(|c| c.context);

        if let Some(context_str) = context_str {
            // Parse the structured error syntax
            let parsed = ParsedError::parse(context_str);
            parsed.to_diagnostic(span)
        } else {
            // No context available, generic error
            DiagnosticBuilder::error(format!("parse error: {:?}", self.code), span.clone())
                .code("E0001")
                .label(span.clone(), "unexpected input")
                .build()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::{IResult, Parser, bytes::complete::tag, error::context};

    type TestResult<'a, T> = IResult<&'a str, T, ContextualError<&'a str>>;

    fn test_parser(input: &str) -> TestResult<&str> {
        context("expected 'hello'", tag("hello")).parse(input)
    }

    #[test]
    fn test_context_capture() {
        let result = test_parser("world");
        match result {
            Err(nom::Err::Error(e)) => {
                assert!(!e.contexts.is_empty());
                assert_eq!(e.contexts[0].context, "expected 'hello'");
                println!("Captured context: {:?}", e.contexts);
            }
            _ => panic!("Expected error with context"),
        }
    }
}
