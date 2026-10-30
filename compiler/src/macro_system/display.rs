//! A display request on the current thread: the editor cursor that
//! `Compiler.getDisplayPos` and `Context.containsDisplayPosition` report to
//! macros, and the `EDisplay` expressions typing reaches, which the editor
//! answers from.

use crate::tast::TypeId;
use std::cell::RefCell;

/// Where a display request asks: a file as macro positions name it, and a
/// character offset into its text, as `Context.getPosInfos` counts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub file: String,
    pub offset: u32,
}

/// What an `EDisplay` asks of the expression it wraps (`haxe.macro.DisplayKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Call,
    Dot,
    Structure,
    Marked,
    Pattern { outermost: bool },
}

/// `EDisplay` lowers to metadata under this reserved name, which no source
/// metadata can spell.
const META_PREFIX: &str = "$display.";

impl Kind {
    pub fn from_variant(name: &str, outermost: bool) -> Option<Self> {
        Some(match name {
            "DKCall" => Kind::Call,
            "DKDot" => Kind::Dot,
            "DKStructure" => Kind::Structure,
            "DKMarked" => Kind::Marked,
            "DKPattern" => Kind::Pattern { outermost },
            _ => return None,
        })
    }

    pub fn variant(self) -> &'static str {
        match self {
            Kind::Call => "DKCall",
            Kind::Dot => "DKDot",
            Kind::Structure => "DKStructure",
            Kind::Marked => "DKMarked",
            Kind::Pattern { .. } => "DKPattern",
        }
    }

    pub fn meta_name(self) -> String {
        match self {
            Kind::Pattern { outermost: true } => format!("{META_PREFIX}DKPattern.outermost"),
            _ => format!("{META_PREFIX}{}", self.variant()),
        }
    }

    pub fn from_meta_name(name: &str) -> Option<Self> {
        let variant = name.strip_prefix(META_PREFIX)?;
        match variant.strip_suffix(".outermost") {
            Some(variant) => Self::from_variant(variant, true),
            None => Self::from_variant(variant, false),
        }
    }
}

/// A display expression typing reached, with the type to complete from: the
/// wrapped expression's, or for a structure the type it is expected to have.
#[derive(Clone, Copy, Debug)]
pub struct Request {
    pub kind: Kind,
    pub type_id: TypeId,
}

/// One task's display request; swapped with `replace_state` when a task
/// sharing the thread is suspended.
#[derive(Default)]
pub struct State {
    cursor: Option<Cursor>,
    requests: Vec<Request>,
}

thread_local! {
    static STATE: RefCell<State> = RefCell::default();
}

/// Exchanges the current thread's display request with a task's saved one.
pub fn replace_state(state: State) -> State {
    STATE.replace(state)
}

/// Asks about `cursor` until dropped, collecting requests afresh.
pub fn at(cursor: Cursor) -> Guard {
    Guard(Some(replace_state(State {
        cursor: Some(cursor),
        requests: Vec::new(),
    })))
}

/// Restores the previous display request when dropped.
pub struct Guard(Option<State>);

impl Guard {
    /// The display expressions typed so far, in the order typing reached them.
    pub fn requests(&self) -> Vec<Request> {
        STATE.with_borrow(|state| state.requests.clone())
    }
}

impl Drop for Guard {
    fn drop(&mut self) {
        if let Some(previous) = self.0.take() {
            replace_state(previous);
        }
    }
}

pub fn cursor() -> Option<Cursor> {
    STATE.with_borrow(|state| state.cursor.clone())
}

pub(crate) fn record(request: Request) {
    STATE.with_borrow_mut(|state| {
        if state.cursor.is_some() {
            state.requests.push(request);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_round_trip_through_their_metadata_names() {
        for kind in [
            Kind::Call,
            Kind::Dot,
            Kind::Structure,
            Kind::Marked,
            Kind::Pattern { outermost: false },
            Kind::Pattern { outermost: true },
        ] {
            assert_eq!(Kind::from_meta_name(&kind.meta_name()), Some(kind));
        }
        assert_eq!(Kind::from_meta_name(":display"), None);
    }

    #[test]
    fn requests_are_collected_only_under_a_cursor() {
        record(Request {
            kind: Kind::Dot,
            type_id: TypeId::from_raw(1),
        });
        let guard = at(Cursor {
            file: "Main.hx".into(),
            offset: 3,
        });
        record(Request {
            kind: Kind::Structure,
            type_id: TypeId::from_raw(2),
        });
        assert_eq!(guard.requests().len(), 1);
        drop(guard);
        assert!(cursor().is_none());
    }
}
