//! Retain encoded responses; allocate JSON trees only for individual requests.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
enum EncodedValue {
    Null,
    Bool(bool),
    Integer(i64),
    Unsigned(u64),
    Float(f64),
    Text(String),
    Array(Vec<EncodedValue>),
    Object(Vec<(String, EncodedValue)>),
    // Semantic token arrays do not need a tag for every integer.
    Integers(Vec<u32>),
}

impl From<Value> for EncodedValue {
    fn from(value: Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(value),
            Value::Number(value) => {
                if let Some(value) = value.as_i64() {
                    Self::Integer(value)
                } else if let Some(value) = value.as_u64() {
                    Self::Unsigned(value)
                } else {
                    Self::Float(value.as_f64().unwrap())
                }
            }
            Value::String(value) => Self::Text(value),
            Value::Array(values) => {
                if values
                    .iter()
                    .all(|value| value.as_u64().is_some_and(|value| value <= u32::MAX as u64))
                {
                    Self::Integers(
                        values
                            .into_iter()
                            .map(|value| value.as_u64().unwrap() as u32)
                            .collect(),
                    )
                } else {
                    Self::Array(values.into_iter().map(Self::from).collect())
                }
            }
            Value::Object(values) => Self::Object(
                values
                    .into_iter()
                    .map(|(key, value)| (key, Self::from(value)))
                    .collect(),
            ),
        }
    }
}

impl EncodedValue {
    fn into_json(self) -> Option<Value> {
        Some(match self {
            Self::Null => Value::Null,
            Self::Bool(value) => value.into(),
            Self::Integer(value) => value.into(),
            Self::Unsigned(value) => value.into(),
            Self::Float(value) => serde_json::Number::from_f64(value)?.into(),
            Self::Text(value) => value.into(),
            Self::Integers(values) => Value::Array(values.into_iter().map(Value::from).collect()),
            Self::Array(values) => Value::Array(
                values
                    .into_iter()
                    .map(Self::into_json)
                    .collect::<Option<_>>()?,
            ),
            Self::Object(values) => Value::Object(
                values
                    .into_iter()
                    .map(|(key, value)| Some((key, value.into_json()?)))
                    .collect::<Option<_>>()?,
            ),
        })
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BinaryResponse(Arc<[u8]>);

impl From<Value> for BinaryResponse {
    fn from(value: Value) -> Self {
        Self(
            postcard::to_allocvec(&EncodedValue::from(value))
                .expect("encodable LSP response")
                .into(),
        )
    }
}

impl BinaryResponse {
    pub fn decode(&self) -> Option<Value> {
        postcard::from_bytes::<EncodedValue>(&self.0)
            .ok()?
            .into_json()
    }

    pub fn heap_bytes(&self) -> usize {
        self.0.len() + 2 * std::mem::size_of::<usize>()
    }
}

impl Serialize for BinaryResponse {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0)
    }
}

impl<'de> Deserialize<'de> for BinaryResponse {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self(Vec::<u8>::deserialize(deserializer)?.into()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
enum Method {
    Hover,
    Signature,
    Definition,
    Highlights,
    References,
    ReferencesWithDeclaration,
    Symbols,
    Tokens,
    Completion,
    InlayHints,
}

impl Method {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "textDocument/hover" => Self::Hover,
            "textDocument/signatureHelp" => Self::Signature,
            "textDocument/definition" => Self::Definition,
            "textDocument/documentHighlight" => Self::Highlights,
            "textDocument/references:false" => Self::References,
            "textDocument/references:true" => Self::ReferencesWithDeclaration,
            "textDocument/documentSymbol" => Self::Symbols,
            "textDocument/semanticTokens/full" => Self::Tokens,
            "textDocument/completion" => Self::Completion,
            "textDocument/inlayHint" => Self::InlayHints,
            _ => return None,
        })
    }
}

#[derive(Clone, Default, Serialize, Deserialize)]
pub(crate) struct Responses(Vec<(Method, bool, BinaryResponse)>);

impl Responses {
    /// A deterministic union of facts from the same source generation.
    pub fn join(&mut self, other: &Self) {
        for (method, typed, response) in &other.0 {
            if let Some((_, old_typed, old)) = self.0.iter_mut().find(|(key, _, _)| key == method) {
                if (*typed, response) > (*old_typed, &*old) {
                    *old_typed = *typed;
                    *old = response.clone();
                }
            } else {
                self.0.push((*method, *typed, response.clone()));
            }
        }
        self.0.sort_by_key(|(method, _, _)| *method);
    }

    /// Preserve provenance per fact when syntax and typed blocks interleave.
    pub fn mark_syntax(&mut self) {
        for (_, typed, _) in &mut self.0 {
            *typed = false;
        }
    }

    pub fn insert(&mut self, method: &str, value: Value) {
        let method = Method::from_name(method).expect("indexed LSP method");
        if let Some((_, typed, old)) = self.0.iter_mut().find(|(key, _, _)| *key == method) {
            *typed = true;
            *old = value.into();
        } else {
            self.0.push((method, true, value.into()));
        }
    }

    pub fn get(&self, method: &str) -> Option<Value> {
        let method = Method::from_name(method)?;
        self.0.iter().find(|(key, _, _)| *key == method)?.2.decode()
    }

    pub fn heap_bytes(&self) -> usize {
        self.0.capacity() * std::mem::size_of::<(Method, bool, BinaryResponse)>()
            + self
                .0
                .iter()
                .map(|(_, _, response)| response.heap_bytes())
                .sum::<usize>()
    }
}

impl<const N: usize> From<[(&str, Value); N]> for Responses {
    fn from(values: [(&str, Value); N]) -> Self {
        let mut responses = Self(Vec::with_capacity(N));
        for (method, value) in values {
            responses.insert(method, value);
        }
        responses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn binary_responses_preserve_wire_values_and_share_storage() {
        let original = json!({"null":null,"boolean":true,"signed":i64::MIN,
            "unsigned":u64::MAX,"float":-0.125,"text":"λ\n文",
            "mixed":[1,"x",{},[]],"tokens":[0,1,127,128,u32::MAX]});
        let encoded = BinaryResponse::from(original.clone());
        let copy = encoded.clone();
        assert!(Arc::ptr_eq(&encoded.0, &copy.0));
        let saved = postcard::to_allocvec(&encoded).unwrap();
        let restored: BinaryResponse = postcard::from_bytes(&saved).unwrap();
        assert_eq!(restored.decode().unwrap(), original);
        assert!(BinaryResponse(Arc::from([255])).decode().is_none());
    }

    #[test]
    fn numeric_token_arrays_stay_compact_in_memory() {
        let encoded = BinaryResponse::from(json!({"data":vec![0u8;600_000]}));
        assert!(encoded.heap_bytes() < 610_000);
        assert_eq!(
            encoded.decode().unwrap()["data"].as_array().unwrap().len(),
            600_000
        );
    }
}
