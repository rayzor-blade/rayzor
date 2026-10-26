//! Binary assets produced and embedded by compile-time generators.

use super::MacroValue;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

pub(crate) type Resources = Arc<Mutex<ResourceStore>>;

#[derive(Default)]
pub(crate) struct ResourceStore {
    configured: Arc<Vec<(String, Vec<u8>)>>,
    added: BTreeMap<String, Vec<u8>>,
    revision: u64,
    embedded: std::collections::BTreeSet<u64>,
}

impl ResourceStore {
    pub(crate) fn configured(resources: Arc<Vec<(String, Vec<u8>)>>) -> Resources {
        Arc::new(Mutex::new(Self {
            configured: resources,
            ..Self::default()
        }))
    }

    pub(crate) fn insert(&mut self, name: String, bytes: Vec<u8>) {
        if self.get(&name) != Some(&bytes) {
            self.added.insert(name, bytes);
            self.revision += 1;
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

    pub(crate) fn mark_embedded(&mut self) {
        self.embedded.insert(self.revision);
    }

    pub(crate) fn has_stale_table(&self) -> bool {
        self.embedded
            .iter()
            .any(|revision| *revision != self.revision)
    }
}

impl std::ops::Deref for ResourceStore {
    type Target = BTreeMap<String, Vec<u8>>;
    fn deref(&self) -> &Self::Target {
        &self.added
    }
}

pub(crate) fn bytes_value(bytes: Vec<u8>) -> MacroValue {
    MacroValue::Object(Arc::new(BTreeMap::from([
        ("__type__".into(), MacroValue::from_str("haxe.io.Bytes")),
        ("length".into(), MacroValue::Int(bytes.len() as i64)),
        (
            "__bytes__".into(),
            MacroValue::Array(Arc::new(
                bytes
                    .into_iter()
                    .map(|byte| MacroValue::Int(byte.into()))
                    .collect(),
            )),
        ),
    ])))
}

pub(crate) fn bytes_data(value: &MacroValue) -> Option<Vec<u8>> {
    let MacroValue::Object(fields) = value else {
        return None;
    };
    if fields.get("__type__").and_then(MacroValue::as_string) != Some("haxe.io.Bytes") {
        return None;
    }
    let MacroValue::Array(bytes) = fields.get("__bytes__")? else {
        return None;
    };
    bytes
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
        Value::Array(values) => {
            MacroValue::Array(Arc::new(values.into_iter().map(json_value).collect()))
        }
        Value::Object(fields) => MacroValue::Object(Arc::new(
            fields
                .into_iter()
                .map(|(name, value)| (name, json_value(value)))
                .collect(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_tables_detect_late_changes_and_keep_configured_bytes() {
        let resources = ResourceStore::configured(Arc::new(vec![("first".into(), vec![1])]));
        let mut resources = resources.lock().unwrap();
        resources.mark_embedded();
        resources.insert("first".into(), vec![1]);
        assert!(!resources.has_stale_table());
        assert_eq!(resources.entries().count(), 1);
        resources.insert("first".into(), vec![2]);
        assert!(resources.has_stale_table());
        assert_eq!(resources.get("first"), Some(&vec![2]));
        assert_eq!(resources.entries().count(), 1);
    }
}
