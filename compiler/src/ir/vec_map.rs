//! An ordered map stored as one sorted vector.
//!
//! The per-function maps of the IR (blocks, register types, locals) are small
//! and mostly built by appending ascending ids. A B-tree spends a whole node,
//! room for eleven entries, on each of them; this map spends one slot per
//! entry. It keeps the B-tree's interface and its order, so iteration is by
//! ascending key either way.

use serde::de::{Deserialize, Deserializer, MapAccess, Visitor};
use serde::ser::{Serialize, SerializeMap, Serializer};
use std::fmt;
use std::marker::PhantomData;

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct VecMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K, V> Default for VecMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: Ord, V> VecMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    fn find(&self, key: &K) -> Result<usize, usize> {
        // Keys mostly arrive in ascending order: check the end first.
        match self.entries.last() {
            Some((last, _)) if last < key => Err(self.entries.len()),
            _ => self.entries.binary_search_by(|(k, _)| k.cmp(key)),
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.find(key).ok().map(|i| &self.entries[i].1)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.find(key).ok().map(|i| &mut self.entries[i].1)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.find(key).is_ok()
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        match self.find(&key) {
            Ok(i) => Some(std::mem::replace(&mut self.entries[i].1, value)),
            Err(i) => {
                self.entries.insert(i, (key, value));
                None
            }
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.find(key).ok().map(|i| self.entries.remove(i).1)
    }

    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        match self.find(&key) {
            Ok(i) => Entry::Occupied(&mut self.entries[i].1),
            Err(i) => Entry::Vacant(self, i, key),
        }
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        self.entries.retain_mut(|(k, v)| keep(k, v));
    }

    pub fn first_key_value(&self) -> Option<(&K, &V)> {
        self.entries.first().map(|(k, v)| (k, v))
    }

    pub fn last_key_value(&self) -> Option<(&K, &V)> {
        self.entries.last().map(|(k, v)| (k, v))
    }

    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator + Clone {
        self.entries.iter().map(|(k, _)| k)
    }

    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator + Clone {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn values_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut V> + ExactSizeIterator {
        self.entries.iter_mut().map(|(_, v)| v)
    }

    pub fn into_values(self) -> impl DoubleEndedIterator<Item = V> + ExactSizeIterator {
        self.entries.into_iter().map(|(_, v)| v)
    }

    pub fn into_keys(self) -> impl DoubleEndedIterator<Item = K> + ExactSizeIterator {
        self.entries.into_iter().map(|(k, _)| k)
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter(self.entries.iter())
    }

    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut(self.entries.iter_mut())
    }

    pub fn shrink_to_fit(&mut self) {
        self.entries.shrink_to_fit();
    }

    /// Sort and keep the last value given for a key, as repeated B-tree
    /// inserts would.
    fn from_unsorted(mut entries: Vec<(K, V)>) -> Self {
        if !entries.windows(2).all(|w| w[0].0 < w[1].0) {
            entries.reverse();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries.dedup_by(|later, earlier| later.0 == earlier.0);
        }
        Self { entries }
    }
}

pub enum Entry<'a, K, V> {
    Occupied(&'a mut V),
    Vacant(&'a mut VecMap<K, V>, usize, K),
}

impl<'a, K: Ord, V> Entry<'a, K, V> {
    pub fn or_insert_with(self, make: impl FnOnce() -> V) -> &'a mut V {
        match self {
            Entry::Occupied(v) => v,
            Entry::Vacant(map, i, key) => {
                map.entries.insert(i, (key, make()));
                &mut map.entries[i].1
            }
        }
    }

    pub fn or_insert(self, value: V) -> &'a mut V {
        self.or_insert_with(|| value)
    }

    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}

#[derive(Clone)]
pub struct Iter<'a, K, V>(std::slice::Iter<'a, (K, V)>);

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (k, v))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for Iter<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(k, v)| (k, v))
    }
}

impl<K, V> ExactSizeIterator for Iter<'_, K, V> {}

pub struct IterMut<'a, K, V>(std::slice::IterMut<'a, (K, V)>);

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(k, v)| (&*k, v))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for IterMut<'_, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.0.next_back().map(|(k, v)| (&*k, v))
    }
}

impl<K, V> ExactSizeIterator for IterMut<'_, K, V> {}

impl<'a, K: Ord, V> IntoIterator for &'a VecMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, K: Ord, V> IntoIterator for &'a mut VecMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<K, V> IntoIterator for VecMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.into_iter()
    }
}

impl<K: Ord, V> FromIterator<(K, V)> for VecMap<K, V> {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        Self::from_unsorted(iter.into_iter().collect())
    }
}

impl<K: Ord, V> Extend<(K, V)> for VecMap<K, V> {
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: Ord, V> std::ops::Index<&K> for VecMap<K, V> {
    type Output = V;
    fn index(&self, key: &K) -> &V {
        self.get(key).expect("no entry found for key")
    }
}

impl<K: fmt::Debug, V: fmt::Debug> fmt::Debug for VecMap<K, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map()
            .entries(self.entries.iter().map(|(k, v)| (k, v)))
            .finish()
    }
}

impl<K: Serialize, V: Serialize> Serialize for VecMap<K, V> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.entries.len()))?;
        for (k, v) in &self.entries {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}

impl<'de, K: Ord + Deserialize<'de>, V: Deserialize<'de>> Deserialize<'de> for VecMap<K, V> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct MapVisitor<K, V>(PhantomData<(K, V)>);
        impl<'de, K: Ord + Deserialize<'de>, V: Deserialize<'de>> Visitor<'de> for MapVisitor<K, V> {
            type Value = VecMap<K, V>;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a map")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut access: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::with_capacity(access.size_hint().unwrap_or(0).min(4096));
                while let Some((k, v)) = access.next_entry()? {
                    entries.push((k, v));
                }
                entries.shrink_to_fit();
                Ok(VecMap::from_unsorted(entries))
            }
        }
        deserializer.deserialize_map(MapVisitor(PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use super::VecMap;

    #[test]
    fn keeps_order_and_last_value() {
        let mut m = VecMap::new();
        m.insert(3, "c");
        m.insert(1, "a");
        m.insert(2, "b");
        assert_eq!(m.insert(1, "A"), Some("a"));
        assert_eq!(m.keys().copied().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(m[&1], "A");
        assert_eq!(m.remove(&2), Some("b"));
        assert!(!m.contains_key(&2));
        let c: VecMap<i32, i32> = vec![(2, 1), (1, 1), (2, 2)].into_iter().collect();
        assert_eq!(
            c.iter().map(|(k, v)| (*k, *v)).collect::<Vec<_>>(),
            vec![(1, 1), (2, 2)]
        );
        *m.entry(5).or_insert("e") = "E";
        assert_eq!(m.get(&5), Some(&"E"));
    }
}
