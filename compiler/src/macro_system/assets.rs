//! Binary assets produced and embedded by compile-time generators.

use super::MacroValue;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) type Resources = Arc<Mutex<ResourceStore>>;

/// The std class `untyped __resources__()` reads its table from. The compiler
/// registers every resource at the start of `haxe.Resource.__init__` once
/// typing is done, so macros may add resources at any point before that.
pub(crate) const RESOURCE_TABLE_CLASS: &str = "rayzor.runtime.Resources";

#[derive(Default)]
pub(crate) struct ResourceStore {
    configured: Arc<Vec<(String, Vec<u8>)>>,
    added: BTreeMap<String, Vec<u8>>,
    /// Every `insert`, including one that changed nothing.
    inserts: u64,
}

impl ResourceStore {
    pub(crate) fn configured(resources: Arc<Vec<(String, Vec<u8>)>>) -> Resources {
        Arc::new(Mutex::new(Self {
            configured: resources,
            ..Self::default()
        }))
    }

    pub(crate) fn insert(&mut self, name: String, bytes: Vec<u8>) {
        self.inserts += 1;
        if self.get(&name) != Some(&bytes) {
            self.added.insert(name, bytes);
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<&Vec<u8>> {
        self.added.get(name).or_else(|| {
            self.configured
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, bytes)| bytes)
        })
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = (&String, &Vec<u8>)> {
        self.configured
            .iter()
            .filter(|(name, _)| !self.added.contains_key(name))
            .map(|(name, bytes)| (name, bytes))
            .chain(self.added.iter())
    }

    /// Every resource, configured and macro-added, with macro bytes winning.
    pub(crate) fn snapshot(&self) -> Vec<(String, Vec<u8>)> {
        self.entries()
            .map(|(name, bytes)| (name.clone(), bytes.clone()))
            .collect()
    }

    /// How many resources macros have offered so far.
    pub(crate) fn inserts(&self) -> u64 {
        self.inserts
    }
}

impl std::ops::Deref for ResourceStore {
    type Target = BTreeMap<String, Vec<u8>>;
    fn deref(&self) -> &Self::Target {
        &self.added
    }
}

pub(crate) fn bytes_value(bytes: Vec<u8>) -> MacroValue {
    MacroValue::object(BTreeMap::from([
        ("__type__".into(), MacroValue::from_str("haxe.io.Bytes")),
        ("length".into(), MacroValue::Int(bytes.len() as i64)),
        (
            "__bytes__".into(),
            MacroValue::array(
                bytes
                    .into_iter()
                    .map(|byte| MacroValue::Int(byte.into()))
                    .collect(),
            ),
        ),
    ]))
}

pub(crate) fn bytes_data(value: &MacroValue) -> Option<Vec<u8>> {
    let MacroValue::Object(fields) = value else {
        return None;
    };
    if fields
        .read()
        .get("__type__")
        .and_then(MacroValue::as_string)
        != Some("haxe.io.Bytes")
    {
        return None;
    }
    let MacroValue::Array(bytes) = fields.read().get("__bytes__").cloned()? else {
        return None;
    };
    bytes
        .read()
        .iter()
        .map(|byte| u8::try_from(byte.as_int()?).ok())
        .collect()
}

pub(crate) fn md5(bytes: &[u8]) -> String {
    use md5::{Digest, Md5};
    format!("{:x}", Md5::digest(bytes))
}

pub(crate) fn json_value(value: serde_json::Value) -> MacroValue {
    use serde_json::Value;
    match value {
        Value::Null => MacroValue::Null,
        Value::Bool(value) => MacroValue::Bool(value),
        Value::String(value) => MacroValue::from_string(value),
        Value::Number(value) => {
            match value.as_i64().filter(|value| i32::try_from(*value).is_ok()) {
                Some(value) => MacroValue::Int(value),
                None => MacroValue::Float(value.as_f64().unwrap()),
            }
        }
        Value::Array(values) => MacroValue::array(values.into_iter().map(json_value).collect()),
        Value::Object(fields) => MacroValue::object(
            fields
                .into_iter()
                .map(|(name, value)| (name, json_value(value)))
                .collect(),
        ),
    }
}

/// Standard Base64 with padding, the form `haxe.crypto.Base64.decode` reads.
pub(crate) fn base64(bytes: &[u8]) -> String {
    const CHARS: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = (u32::from(b[0]) << 16) | (u32::from(b[1]) << 8) | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(CHARS[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_bytes_replace_configured_bytes() {
        let resources = ResourceStore::configured(Arc::new(vec![("first".into(), vec![1])]));
        let mut resources = resources.lock().unwrap();
        resources.insert("first".into(), vec![1]);
        assert_eq!(resources.entries().count(), 1);
        resources.insert("first".into(), vec![2]);
        assert_eq!(resources.get("first"), Some(&vec![2]));
        assert_eq!(resources.entries().count(), 1);
    }

    #[test]
    fn base64_pads_partial_groups() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(
            base64("H\u{e9}llo World !".as_bytes()),
            "SMOpbGxvIFdvcmxkICE="
        );
    }
}
