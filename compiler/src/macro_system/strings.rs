//! Bounded character indexes for repeated macro-time string reads.

use std::{cell::RefCell, collections::VecDeque, mem::size_of, sync::Arc};

const MAX_BYTES: usize = 8 << 20;
const MAX_STRINGS: usize = 16;

struct Entry {
    source: Arc<str>,
    characters: Vec<char>,
}

impl Entry {
    fn bytes(&self) -> usize {
        self.source.len() + self.characters.capacity() * size_of::<char>()
    }
}

#[derive(Default)]
pub(super) struct Characters(RefCell<VecDeque<Entry>>);

impl Characters {
    pub fn at(&self, text: &Arc<str>, index: usize) -> Option<char> {
        self.cached(text, |characters| characters.get(index).copied())
            .unwrap_or_else(|| text.chars().nth(index))
    }

    pub fn len(&self, text: &Arc<str>) -> usize {
        self.cached(text, <[char]>::len)
            .unwrap_or_else(|| text.chars().count())
    }

    fn cached<R>(&self, text: &Arc<str>, query: impl FnOnce(&[char]) -> R) -> Option<R> {
        if text.len() > MAX_BYTES / (1 + size_of::<char>()) {
            return None;
        }
        let mut entries = self.0.borrow_mut();
        let entry = match entries
            .iter()
            .position(|entry| Arc::ptr_eq(&entry.source, text))
        {
            Some(index) => entries.remove(index).unwrap(),
            None => Entry {
                source: text.clone(),
                characters: text.chars().collect(),
            },
        };
        if entry.bytes() > MAX_BYTES {
            return None;
        }
        let result = query(&entry.characters);
        entries.push_back(entry);
        while entries.len() > MAX_STRINGS
            || entries.iter().map(Entry::bytes).sum::<usize>() > MAX_BYTES
        {
            entries.pop_front();
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn character_reads_preserve_unicode_and_do_not_retain_every_input() {
        let cache = Characters::default();
        let text: Arc<str> = Arc::from("aé🙂z");
        for _ in 0..1000 {
            assert_eq!(cache.len(&text), 4);
            assert_eq!(cache.at(&text, 2), Some('🙂'));
            assert_eq!(cache.at(&text, 4), None);
        }
        for index in 0..100 {
            let value: Arc<str> = Arc::from(format!("{index}{}", "é".repeat(100_000)));
            assert_eq!(cache.at(&value, index.to_string().len()), Some('é'));
        }
        let entries = cache.0.borrow();
        assert!(entries.len() <= MAX_STRINGS);
        assert!(entries.iter().map(Entry::bytes).sum::<usize>() <= MAX_BYTES);
    }
}
