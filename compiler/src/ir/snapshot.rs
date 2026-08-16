//! The standard library, lowered at build time and carried in the binary.
//!
//! Compiling a program means lowering whatever part of the standard library it
//! reaches. That work is the same for every program, so the CLI's build script
//! does it once and stores the result here; a compile looks a module up instead
//! of lowering it again, and a project that has never been built is no slower
//! than one that has.
//!
//! The archive is a flat sequence of length-prefixed entries, read in place
//! from the binary's own data rather than parsed into owned buffers:
//!
//! ```text
//! magic "RZSN" | u32 count | [ u32 key_len | key | u32 data_len | data ]*
//! ```
//!
//! Keys are `<cache-discriminator>/<module>.blade`, matching where the
//! project cache would have kept the same artifact, so a lookup asks the same
//! question of both.

use std::collections::BTreeMap;

pub const SNAPSHOT_MAGIC: &[u8; 4] = b"RZSN";

/// Build an archive from `(key, artifact bytes)` pairs.
pub fn encode(entries: &BTreeMap<String, Vec<u8>>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(SNAPSHOT_MAGIC);
    out.extend_from_slice(&(entries.len() as u32).to_le_bytes());
    for (key, data) in entries {
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key.as_bytes());
        out.extend_from_slice(&(data.len() as u32).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// Index an archive without copying it. Entries borrow from `bytes`, so a
/// snapshot embedded in the binary is indexed straight out of its own data.
///
/// A malformed archive yields an empty index rather than an error: the
/// snapshot is an optimisation, and a compile that cannot use it still lowers
/// from source.
pub fn index(bytes: &[u8]) -> BTreeMap<&str, &[u8]> {
    let mut map = BTreeMap::new();
    if bytes.len() < 8 || &bytes[..4] != SNAPSHOT_MAGIC {
        return map;
    }
    let count = u32::from_le_bytes(match bytes[4..8].try_into() {
        Ok(v) => v,
        Err(_) => return map,
    });

    let mut pos = 8usize;
    for _ in 0..count {
        let Some(key) = read_slice(bytes, &mut pos) else {
            return map;
        };
        let Some(data) = read_slice(bytes, &mut pos) else {
            return map;
        };
        let Ok(key) = std::str::from_utf8(key) else {
            return map;
        };
        map.insert(key, data);
    }
    map
}

fn read_slice<'a>(bytes: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
    let start = *pos;
    let len_end = start.checked_add(4)?;
    let len = u32::from_le_bytes(bytes.get(start..len_end)?.try_into().ok()?) as usize;
    let end = len_end.checked_add(len)?;
    let slice = bytes.get(len_end..end)?;
    *pos = end;
    Some(slice)
}

/// The key a module occupies in the archive.
pub fn key_for(discriminator: &str, module_file: &str) -> String {
    format!("{discriminator}/{module_file}")
}

/// The archive this binary carries, staged into `OUT_DIR` by this crate's
/// build script.
///
/// It belongs to the compiler rather than to whichever binary embeds the
/// compiler: every host that lowers Haxe wants the same prepared library, and
/// making each one find the archive and hand it over at startup is how a host
/// ends up silently compiling without it. A build with nothing to stage
/// embeds an empty archive, which reads as "carries nothing" and falls back to
/// lowering from source.
static ARCHIVE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/stdlib_snapshot.bin"));

/// The standard library this binary carries, indexed on first use.
pub fn installed() -> &'static BTreeMap<&'static str, &'static [u8]> {
    static INDEXED: std::sync::OnceLock<BTreeMap<&'static str, &'static [u8]>> =
        std::sync::OnceLock::new();
    INDEXED.get_or_init(|| index(ARCHIVE))
}
