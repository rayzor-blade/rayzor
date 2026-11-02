//! Filesystem queries available to compile-time asset generators.

use super::{MacroError, MacroValue};
use crate::tast::SourceLocation;
use std::sync::Arc;

pub(super) fn call(
    method: &str,
    args: &[MacroValue],
    location: SourceLocation,
) -> Result<Option<MacroValue>, MacroError> {
    if !matches!(
        method,
        "exists"
            | "isDirectory"
            | "fullPath"
            | "absolutePath"
            | "readDirectory"
            | "createDirectory"
            | "deleteFile"
            | "deleteDirectory"
            | "rename"
            | "stat"
    ) {
        return Ok(None);
    }
    let path = args
        .first()
        .and_then(MacroValue::as_string)
        .ok_or_else(|| MacroError::TypeError {
            message: format!("FileSystem.{method} expects a path string"),
            location,
        })?;
    let fail = |error: std::io::Error| MacroError::RuntimeError {
        message: format!("FileSystem.{method}('{path}') failed: {error}"),
        location,
    };
    let string =
        |path: std::path::PathBuf| MacroValue::String(Arc::from(path.to_string_lossy().as_ref()));
    Ok(Some(match method {
        "exists" => MacroValue::Bool(std::path::Path::new(path).exists()),
        "isDirectory" => MacroValue::Bool(std::path::Path::new(path).is_dir()),
        "createDirectory" => {
            super::budget::require_effects(location)?;
            std::fs::create_dir_all(path).map_err(fail)?;
            MacroValue::Null
        }
        "deleteFile" => {
            super::budget::require_effects(location)?;
            std::fs::remove_file(path).map_err(fail)?;
            MacroValue::Null
        }
        "deleteDirectory" => {
            super::budget::require_effects(location)?;
            std::fs::remove_dir(path).map_err(fail)?;
            MacroValue::Null
        }
        "rename" => {
            super::budget::require_effects(location)?;
            let target = args.get(1).and_then(MacroValue::as_string).ok_or_else(|| {
                MacroError::TypeError {
                    message: "FileSystem.rename expects a new path string".into(),
                    location,
                }
            })?;
            std::fs::rename(path, target).map_err(fail)?;
            MacroValue::Null
        }
        "stat" => stat(&std::fs::metadata(path).map_err(fail)?),
        "fullPath" => string(std::fs::canonicalize(path).map_err(fail)?),
        "absolutePath" if std::path::Path::new(path).is_absolute() => {
            MacroValue::String(Arc::from(path))
        }
        "absolutePath" => {
            let absolute =
                std::path::absolute(if path.is_empty() { "." } else { path }).map_err(fail)?;
            let mut normalized = std::path::PathBuf::new();
            for component in absolute.components() {
                match component {
                    std::path::Component::ParentDir => {
                        normalized.pop();
                    }
                    std::path::Component::CurDir => {}
                    component => normalized.push(component.as_os_str()),
                }
            }
            string(normalized)
        }
        "readDirectory" => {
            let entries = std::fs::read_dir(path)
                .map_err(fail)?
                .map(|entry| {
                    entry
                        .map(|entry| {
                            MacroValue::String(Arc::from(
                                entry.file_name().to_string_lossy().as_ref(),
                            ))
                        })
                        .map_err(fail)
                })
                .collect::<Result<Vec<_>, _>>()?;
            MacroValue::array(entries)
        }
        _ => unreachable!(),
    }))
}

/// A `sys.FileStat`; its times are `Date` objects, and ids are zero where the host has none.
fn stat(metadata: &std::fs::Metadata) -> MacroValue {
    let date = |time: std::io::Result<std::time::SystemTime>| {
        super::interpreter::host_std::date_value(
            time.ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0.0, |elapsed| elapsed.as_millis() as f64),
        )
    };
    #[cfg(unix)]
    let ids = {
        use std::os::unix::fs::MetadataExt;
        [
            u64::from(metadata.gid()),
            u64::from(metadata.uid()),
            metadata.dev(),
            metadata.ino(),
            metadata.nlink(),
            metadata.rdev(),
            u64::from(metadata.mode()),
        ]
    };
    #[cfg(not(unix))]
    let ids = [0u64; 7];
    let [gid, uid, dev, ino, nlink, rdev, mode] = ids.map(|id| MacroValue::Int(id as i64));
    MacroValue::object(
        [
            ("gid", gid),
            ("uid", uid),
            ("atime", date(metadata.accessed())),
            ("mtime", date(metadata.modified())),
            (
                "ctime",
                date(metadata.created().or_else(|_| metadata.modified())),
            ),
            ("size", MacroValue::Int(metadata.len() as i64)),
            ("dev", dev),
            ("ino", ino),
            ("nlink", nlink),
            ("rdev", rdev),
            ("mode", mode),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_string(), value))
        .collect(),
    )
}
