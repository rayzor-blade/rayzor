//! Bounded character indexes for repeated macro-time string reads.

use std::{cell::RefCell, collections::VecDeque, mem::size_of, sync::Arc};

const MAX_BYTES: usize = 8 << 20;
const MAX_STRINGS: usize = 16;

pub(super) fn search(text: &str, needle: &str, start: Option<i64>, reverse: bool) -> i64 {
    let length = text.chars().count();
    let start = start
        .map(|index| index.max(0) as usize)
        .unwrap_or(if reverse { length } else { 0 });
    if !reverse && start > length {
        return -1;
    }
    let start = start.min(length);
    if needle.is_empty() {
        return start as i64;
    }
    if reverse {
        text.char_indices()
            .enumerate()
            .take(start + 1)
            .filter_map(|(index, (offset, _))| {
                text[offset..].starts_with(needle).then_some(index as i64)
            })
            .last()
            .unwrap_or(-1)
    } else {
        let offset = text
            .char_indices()
            .nth(start)
            .map_or(text.len(), |(offset, _)| offset);
        text[offset..].find(needle).map_or(-1, |found| {
            (start + text[offset..offset + found].chars().count()) as i64
        })
    }
}

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
    fn search_preserves_character_offsets_bounds_and_overlapping_matches() {
        for (text, needle, start, reverse, expected) in [
            ("aé🙂é", "é", None, false, 1),
            ("aé🙂é", "é", Some(2), false, 3),
            ("aé🙂é", "é", None, true, 3),
            ("aé🙂é", "é", Some(2), true, 1),
            ("ababa", "aba", None, true, 2),
            ("ababa", "aba", Some(1), true, 0),
            ("abc", "", Some(3), false, 3),
            ("abc", "", Some(4), false, -1),
            ("abc", "", None, true, 3),
            ("abc", "z", None, true, -1),
        ] {
            assert_eq!(
                search(text, needle, start, reverse),
                expected,
                "{text} {needle} {start:?} {reverse}"
            );
        }
    }

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
