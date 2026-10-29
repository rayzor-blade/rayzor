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
            MacroValue::Array(Arc::new(entries))
        }
        _ => unreachable!(),
    }))
}
