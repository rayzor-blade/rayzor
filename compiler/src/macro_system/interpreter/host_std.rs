//! Host implementations of the extern standard library that build macros call:
//! `Std`, `Math`, `Sys`, `Date`, `sys.io.File` streams, `haxe.Json`, `haxe.io.Bytes`,
//! `haxe.crypto` and `Type`.

use super::*;
use chrono::{Datelike, TimeZone, Timelike};
use std::sync::Mutex;

/// `Sys.putEnv` values; `None` removes a variable. They reach `Sys.getEnv`,
/// `Sys.environment` and child processes without changing the compiler's own environment.
static ENVIRONMENT: Mutex<BTreeMap<String, Option<String>>> = Mutex::new(BTreeMap::new());

pub(super) fn env_var(name: &str) -> Option<String> {
    match ENVIRONMENT.lock().unwrap().get(name) {
        Some(value) => value.clone(),
        None => std::env::var(name).ok(),
    }
}

/// Applies `Sys.putEnv` values to a child process.
pub(super) fn apply_environment(command: &mut std::process::Command) {
    for (name, value) in ENVIRONMENT.lock().unwrap().iter() {
        match value {
            Some(value) => command.env(name, value),
            None => command.env_remove(name),
        };
    }
}

/// A counter mixed with the clock: macros use it to name things apart, not for secrets.
fn random_bits() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static STATE: AtomicU64 = AtomicU64::new(0);
    let clock = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos() as u64);
    let mut z = STATE
        .fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
        .wrapping_add(clock);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A `Date` at `milliseconds` since the Unix epoch.
pub(crate) fn date_value(milliseconds: f64) -> MacroValue {
    MacroValue::object(BTreeMap::from([
        ("__type__".to_string(), MacroValue::from_str("Date")),
        ("__time__".to_string(), MacroValue::Float(milliseconds)),
    ]))
}

fn local_time(milliseconds: f64) -> Option<chrono::DateTime<chrono::Local>> {
    chrono::Local
        .timestamp_millis_opt(milliseconds as i64)
        .single()
}

/// `Date.toString()`: local time as `YYYY-MM-DD hh:mm:ss`.
pub(crate) fn date_string(milliseconds: f64) -> String {
    local_time(milliseconds).map_or_else(
        || "Invalid Date".to_string(),
        |time| time.format("%Y-%m-%d %H:%M:%S").to_string(),
    )
}

fn local_milliseconds(time: chrono::NaiveDateTime) -> Option<f64> {
    chrono::Local
        .from_local_datetime(&time)
        .earliest()
        .map(|time| time.timestamp_millis() as f64)
}

fn eof(location: SourceLocation) -> MacroError {
    MacroError::Thrown {
        value: Box::new(MacroValue::object(BTreeMap::from([(
            "__type__".to_string(),
            MacroValue::from_str("haxe.io.Eof"),
        )]))),
        location,
    }
}

fn int_arg(args: &[MacroValue], index: usize) -> Option<i64> {
    args.get(index).and_then(MacroValue::as_int)
}

fn string_arg<'a>(
    args: &'a [MacroValue],
    index: usize,
    operation: &str,
    location: SourceLocation,
) -> Result<&'a str, MacroError> {
    args.get(index)
        .and_then(MacroValue::as_string)
        .ok_or_else(|| MacroError::TypeError {
            message: format!("{operation} expects a string argument"),
            location,
        })
}

fn bytes_arg(
    args: &[MacroValue],
    index: usize,
    operation: &str,
    location: SourceLocation,
) -> Result<Vec<u8>, MacroError> {
    args.get(index)
        .and_then(super::super::assets::bytes_data)
        .ok_or_else(|| MacroError::TypeError {
            message: format!("{operation} expects Bytes"),
            location,
        })
}

/// The `__bytes__` storage of a `haxe.io.Bytes`, shared so writes land in place.
fn bytes_storage(value: &MacroValue) -> Option<Shared<Vec<MacroValue>>> {
    match value {
        MacroValue::Object(fields) => match fields.read().get("__bytes__") {
            Some(MacroValue::Array(storage)) => Some(storage.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn byte_at(storage: &[MacroValue], index: usize) -> u8 {
    storage.get(index).and_then(MacroValue::as_int).unwrap_or(0) as u8
}

/// `pos..pos + len` inside `0..length`, as Haxe's bounds checks require.
fn checked_range(
    pos: i64,
    len: i64,
    length: usize,
    operation: &str,
    location: SourceLocation,
) -> Result<std::ops::Range<usize>, MacroError> {
    if pos < 0 || len < 0 || (pos + len) as usize > length {
        return Err(MacroError::RuntimeError {
            message: format!("{operation}: out of bounds"),
            location,
        });
    }
    Ok(pos as usize..(pos + len) as usize)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Writes `value` as `JsonPrinter` does: fields of anonymous objects and maps, enums
/// as their index, dates as their string, and `space` indentation when given.
struct JsonWriter<'a> {
    replacer: Option<MacroValue>,
    indent: Option<&'a str>,
    depth: usize,
    out: String,
}

impl JsonWriter<'_> {
    fn newline(&mut self) {
        if let Some(indent) = self.indent {
            self.out.push('\n');
            for _ in 0..self.depth {
                self.out.push_str(indent);
            }
        }
    }

    fn quote(&mut self, text: &str) {
        self.out.push('"');
        for character in text.chars() {
            match character {
                '"' => self.out.push_str("\\\""),
                '\\' => self.out.push_str("\\\\"),
                '\n' => self.out.push_str("\\n"),
                '\r' => self.out.push_str("\\r"),
                '\t' => self.out.push_str("\\t"),
                '\u{8}' => self.out.push_str("\\b"),
                '\u{c}' => self.out.push_str("\\f"),
                other => self.out.push(other),
            }
        }
        self.out.push('"');
    }

    fn fields(
        &mut self,
        interpreter: &mut MacroInterpreter,
        fields: Vec<(String, MacroValue)>,
        location: SourceLocation,
    ) -> Result<(), MacroError> {
        self.out.push('{');
        let mut empty = true;
        for (name, value) in fields {
            if matches!(value, MacroValue::Function(_)) {
                continue;
            }
            if empty {
                self.depth += 1;
                empty = false;
            } else {
                self.out.push(',');
            }
            self.newline();
            self.quote(&name);
            self.out.push(':');
            if self.indent.is_some() {
                self.out.push(' ');
            }
            self.write(interpreter, MacroValue::from_str(&name), value, location)?;
        }
        if !empty {
            self.depth -= 1;
            self.newline();
        }
        self.out.push('}');
        Ok(())
    }

    fn write(
        &mut self,
        interpreter: &mut MacroInterpreter,
        key: MacroValue,
        value: MacroValue,
        location: SourceLocation,
    ) -> Result<(), MacroError> {
        let value = match self.replacer.clone() {
            Some(replacer) => interpreter.call_value(replacer, vec![key, value], location)?,
            None => value,
        };
        match value.underlying().clone() {
            MacroValue::Null => self.out.push_str("null"),
            MacroValue::Bool(value) => self.out.push_str(if value { "true" } else { "false" }),
            MacroValue::Int(value) => self.out.push_str(&value.to_string()),
            MacroValue::Float(value) if !value.is_finite() => self.out.push_str("null"),
            MacroValue::Float(value) if value.fract() == 0.0 && value.abs() < 1e21 => {
                self.out.push_str(&format!("{value:.0}"))
            }
            MacroValue::Float(value) => self.out.push_str(&value.to_string()),
            MacroValue::String(text) => self.quote(&text),
            MacroValue::Function(_) => self.out.push_str("\"<fun>\""),
            MacroValue::Array(items) => {
                let items = items.read();
                self.out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        self.out.push(',');
                    } else {
                        self.depth += 1;
                    }
                    self.newline();
                    self.write(
                        interpreter,
                        MacroValue::Int(index as i64),
                        item.clone(),
                        location,
                    )?;
                    if index + 1 == items.len() {
                        self.depth -= 1;
                        self.newline();
                    }
                }
                self.out.push(']');
            }
            MacroValue::Map(entries) => {
                let fields = entries
                    .read()
                    .iter()
                    .map(|(key, value)| (key.to_display_string(), value.clone()))
                    .collect();
                self.fields(interpreter, fields, location)?;
            }
            MacroValue::Object(fields) => {
                let fields = fields.read();
                if let Some("Date") = fields.get("__type__").and_then(MacroValue::as_string) {
                    let time = fields.get("__time__").and_then(MacroValue::as_float);
                    self.quote(&date_string(time.unwrap_or(0.0)));
                    return Ok(());
                }
                let fields = fields
                    .iter()
                    .filter(|(name, _)| !(name.starts_with("__") && name.ends_with("__")))
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                self.fields(interpreter, fields, location)?;
            }
            MacroValue::Enum(owner, name, _) => {
                let index = interpreter.enum_index(&owner, &name, location)?;
                self.out.push_str(&index.to_string());
            }
            other => {
                return Err(MacroError::TypeError {
                    message: format!("haxe.Json.stringify cannot write {}", other.type_name()),
                    location,
                });
            }
        }
        Ok(())
    }
}

impl MacroInterpreter {
    /// Static members of the extern std that the macro host supplies; `None` leaves the
    /// call to the other host handlers.
    pub(super) fn host_std_static(
        &mut self,
        class: &str,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        let value = match (class, method) {
            ("Std", "random") => {
                let max = int_arg(args, 0).unwrap_or(0);
                MacroValue::Int(if max > 0 {
                    (random_bits() % max as u64) as i64
                } else {
                    0
                })
            }
            ("Std", "downcast" | "instance") => {
                let value = args.first().cloned().unwrap_or(MacroValue::Null);
                let class = args.get(1).cloned().unwrap_or(MacroValue::Null);
                if self.is_of_type(&value, &class) {
                    value
                } else {
                    MacroValue::Null
                }
            }
            ("Math", "random") => {
                MacroValue::Float((random_bits() >> 11) as f64 / (1u64 << 53) as f64)
            }
            ("Date", "now") => date_value(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0.0, |elapsed| elapsed.as_millis() as f64),
            ),
            ("Date", "fromTime") => {
                date_value(args.first().and_then(MacroValue::as_float).unwrap_or(0.0))
            }
            ("Date", "fromString") => {
                let text = string_arg(args, 0, "Date.fromString", location)?;
                let invalid = || MacroError::RuntimeError {
                    message: format!("Date.fromString: invalid date format '{text}'"),
                    location,
                };
                let milliseconds = match text.len() {
                    8 => chrono::NaiveTime::parse_from_str(text, "%H:%M:%S")
                        .map(|time| time.num_seconds_from_midnight() as f64 * 1000.0)
                        .ok(),
                    10 => chrono::NaiveDate::parse_from_str(text, "%Y-%m-%d")
                        .ok()
                        .and_then(|date| date.and_hms_opt(0, 0, 0))
                        .and_then(local_milliseconds),
                    19 => chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S")
                        .ok()
                        .and_then(local_milliseconds),
                    _ => None,
                };
                date_value(milliseconds.ok_or_else(invalid)?)
            }
            ("Sys" | "sys.Sys", _) => return self.host_system(method, args, location),
            ("sys.io.File", "saveBytes" | "copy" | "write" | "append" | "read" | "update") => {
                return self.host_file(method, args, location).map(Some);
            }
            ("haxe.Json" | "haxe.format.JsonPrinter", "stringify" | "print") => {
                let replacer = args
                    .get(1)
                    .filter(|value| !matches!(value, MacroValue::Null))
                    .cloned();
                let indent = args.get(2).and_then(MacroValue::as_string);
                let mut writer = JsonWriter {
                    replacer,
                    indent,
                    depth: 0,
                    out: String::new(),
                };
                let value = args.first().cloned().unwrap_or(MacroValue::Null);
                writer.write(self, MacroValue::from_str(""), value, location)?;
                MacroValue::from_string(writer.out)
            }
            ("haxe.io.Bytes", "alloc") => {
                let length = int_arg(args, 0)
                    .filter(|length| *length >= 0)
                    .ok_or_else(|| MacroError::TypeError {
                        message: "Bytes.alloc expects a length".into(),
                        location,
                    })?;
                super::super::assets::bytes_value(vec![0; length as usize])
            }
            ("haxe.io.Bytes", "ofHex") => {
                let text = string_arg(args, 0, "Bytes.ofHex", location)?;
                let digits = text.as_bytes();
                let invalid = || MacroError::RuntimeError {
                    message: format!("Bytes.ofHex: not hexadecimal '{text}'"),
                    location,
                };
                if digits.len() % 2 != 0 {
                    return Err(invalid());
                }
                let bytes = digits
                    .chunks(2)
                    .map(|pair| {
                        std::str::from_utf8(pair)
                            .ok()
                            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
                    })
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(invalid)?;
                super::super::assets::bytes_value(bytes)
            }
            ("haxe.io.Bytes", "fastGet") => {
                let storage = args.first().and_then(bytes_storage);
                let index = int_arg(args, 1).unwrap_or(0).max(0) as usize;
                MacroValue::Int(
                    storage
                        .map_or(0, |storage| byte_at(&storage.read(), index))
                        .into(),
                )
            }
            (
                "haxe.crypto.Sha1" | "haxe.crypto.Sha224" | "haxe.crypto.Sha256",
                "encode" | "make",
            ) => {
                use sha1::Digest;
                let input = if method == "encode" {
                    string_arg(args, 0, class, location)?.as_bytes().to_vec()
                } else {
                    bytes_arg(args, 0, class, location)?
                };
                let digest = match class {
                    "haxe.crypto.Sha1" => sha1::Sha1::digest(&input).to_vec(),
                    "haxe.crypto.Sha224" => sha2::Sha224::digest(&input).to_vec(),
                    _ => sha2::Sha256::digest(&input).to_vec(),
                };
                if method == "encode" {
                    MacroValue::from_string(hex(&digest))
                } else {
                    super::super::assets::bytes_value(digest)
                }
            }
            ("haxe.crypto.Base64", "encode" | "urlEncode") => {
                use base64::Engine;
                let bytes = bytes_arg(args, 0, "Base64.encode", location)?;
                let padded = match args.get(1) {
                    Some(MacroValue::Bool(complement)) => *complement,
                    _ => method == "encode",
                };
                let engine = match (method, padded) {
                    ("encode", true) => base64::engine::general_purpose::STANDARD,
                    ("encode", false) => base64::engine::general_purpose::STANDARD_NO_PAD,
                    (_, true) => base64::engine::general_purpose::URL_SAFE,
                    (_, false) => base64::engine::general_purpose::URL_SAFE_NO_PAD,
                };
                MacroValue::from_string(engine.encode(bytes))
            }
            ("haxe.crypto.Base64", "decode" | "urlDecode") => {
                use base64::Engine;
                use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
                let text = string_arg(args, 0, "Base64.decode", location)?;
                let config = GeneralPurposeConfig::new()
                    .with_decode_padding_mode(DecodePaddingMode::Indifferent)
                    .with_decode_allow_trailing_bits(true);
                let alphabet = if method == "decode" {
                    &base64::alphabet::STANDARD
                } else {
                    &base64::alphabet::URL_SAFE
                };
                let bytes =
                    GeneralPurpose::new(alphabet, config)
                        .decode(text)
                        .map_err(|error| MacroError::RuntimeError {
                            message: format!("Base64.{method}: {error}"),
                            location,
                        })?;
                super::super::assets::bytes_value(bytes)
            }
            ("haxe.crypto.Crc32", "make") => {
                MacroValue::Int(crc32(&bytes_arg(args, 0, "Crc32.make", location)?) as i32 as i64)
            }
            ("haxe.crypto.Adler32", "make") => {
                MacroValue::Int(
                    adler32(&bytes_arg(args, 0, "Adler32.make", location)?) as i32 as i64,
                )
            }
            ("Type", _) => return self.host_type(method, args, location),
            ("Reflect", "compareMethods") => MacroValue::Bool(args.first() == args.get(1)),
            _ => return Ok(None),
        };
        Ok(Some(value))
    }

    fn host_system(
        &mut self,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        let unsupported = |reason: &str| MacroError::UnsupportedOperation {
            operation: format!("Sys.{method} in a macro: {reason}"),
            location,
        };
        Ok(Some(match method {
            "getEnv" => {
                let name = string_arg(args, 0, "Sys.getEnv", location)?;
                env_var(name).map_or(MacroValue::Null, MacroValue::from_string)
            }
            "putEnv" => {
                super::super::budget::require_effects(location)?;
                let name = string_arg(args, 0, "Sys.putEnv", location)?.to_string();
                let value = args
                    .get(1)
                    .and_then(MacroValue::as_string)
                    .map(str::to_string);
                ENVIRONMENT.lock().unwrap().insert(name, value);
                MacroValue::Null
            }
            "environment" => {
                let mut variables: BTreeMap<String, String> = std::env::vars().collect();
                for (name, value) in ENVIRONMENT.lock().unwrap().iter() {
                    match value {
                        Some(value) => variables.insert(name.clone(), value.clone()),
                        None => variables.remove(name),
                    };
                }
                MacroValue::map(
                    variables
                        .into_iter()
                        .map(|(name, value)| {
                            (
                                MacroValue::from_string(name),
                                MacroValue::from_string(value),
                            )
                        })
                        .collect(),
                )
            }
            // A macro runs inside the compiler, which has no program arguments to give it.
            "args" => MacroValue::array(Vec::new()),
            "cpuArch" => MacroValue::from_str(match std::env::consts::ARCH {
                "aarch64" => "arm64",
                "x86" => "x86",
                "x86_64" => "x86_64",
                "arm" => "arm",
                other => other,
            }),
            "cpuTime" => MacroValue::Float(
                cpu_time()
                    .ok_or_else(|| unsupported("this host does not report process CPU time"))?,
            ),
            "timestamp_ms" => MacroValue::Int(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |elapsed| elapsed.as_millis() as i64),
            ),
            "sleep" => {
                let seconds = args.first().and_then(MacroValue::as_float).unwrap_or(0.0);
                if seconds > 0.0 {
                    std::thread::sleep(std::time::Duration::from_secs_f64(seconds));
                }
                MacroValue::Null
            }
            "setTimeLocale" => MacroValue::Bool(false),
            "exit" => {
                return Err(MacroError::RuntimeError {
                    message: format!(
                        "Sys.exit({}) ended the macro",
                        int_arg(args, 0).unwrap_or(0)
                    ),
                    location,
                });
            }
            "stdout" | "stderr" => MacroValue::object(BTreeMap::from([(
                "__type__".to_string(),
                MacroValue::from_str(if method == "stdout" {
                    "sys.io.Stdout"
                } else {
                    "sys.io.Stderr"
                }),
            )])),
            "stdin" | "getChar" => {
                return Err(unsupported(
                    "standard input carries the compiler's own protocol",
                ));
            }
            "setCwd" => {
                return Err(unsupported(
                    "the working directory is shared by every compilation in the process",
                ));
            }
            _ => return self.system_call(method, args, location),
        }))
    }

    fn host_file(
        &mut self,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let operation = format!("File.{method}");
        let path = string_arg(args, 0, &operation, location)?.to_string();
        let fail = |error: std::io::Error| MacroError::RuntimeError {
            message: format!("{operation}('{path}') failed: {error}"),
            location,
        };
        match method {
            "saveBytes" => {
                super::super::budget::require_effects(location)?;
                let bytes = bytes_arg(args, 1, &operation, location)?;
                std::fs::write(&path, bytes).map_err(fail)?;
                if let Some(dependencies) = self.dependencies() {
                    dependencies.refresh_written(&path);
                }
                Ok(MacroValue::Null)
            }
            "copy" => {
                super::super::budget::require_effects(location)?;
                let target = string_arg(args, 1, &operation, location)?;
                if let Some(dependencies) = self.dependencies() {
                    dependencies.register_read(&path);
                }
                std::fs::copy(&path, target).map_err(fail)?;
                if let Some(dependencies) = self.dependencies() {
                    dependencies.refresh_written(target);
                }
                Ok(MacroValue::Null)
            }
            "read" => {
                if let Some(dependencies) = self.dependencies() {
                    dependencies.register_read(&path);
                }
                let bytes = std::fs::read(&path).map_err(fail)?;
                if let Some(dependencies) = self.dependencies() {
                    dependencies.register_content(&path, &bytes);
                }
                Ok(MacroValue::object(BTreeMap::from([
                    (
                        "__type__".to_string(),
                        MacroValue::from_str("sys.io.FileInput"),
                    ),
                    ("__path__".to_string(), MacroValue::from_string(path)),
                    (
                        "__bytes__".to_string(),
                        MacroValue::array(
                            bytes
                                .into_iter()
                                .map(|byte| MacroValue::Int(byte.into()))
                                .collect(),
                        ),
                    ),
                    ("__pos__".to_string(), MacroValue::Int(0)),
                    ("bigEndian".to_string(), MacroValue::Bool(false)),
                ])))
            }
            "write" | "append" => {
                super::super::budget::require_effects(location)?;
                let mut options = std::fs::OpenOptions::new();
                if method == "write" {
                    options.write(true).create(true).truncate(true);
                } else {
                    options.append(true).create(true);
                }
                options.open(&path).map_err(fail)?;
                if let Some(dependencies) = self.dependencies() {
                    dependencies.refresh_written(&path);
                }
                Ok(MacroValue::object(BTreeMap::from([
                    (
                        "__type__".to_string(),
                        MacroValue::from_str("sys.io.FileOutput"),
                    ),
                    ("__path__".to_string(), MacroValue::from_string(path)),
                    ("__buffer__".to_string(), MacroValue::array(Vec::new())),
                    ("bigEndian".to_string(), MacroValue::Bool(false)),
                ])))
            }
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("{operation} in a macro: use File.read and File.write"),
                location,
            }),
        }
    }

    /// The constructor's position in its enum's declaration.
    pub(super) fn enum_index(
        &self,
        owner: &str,
        name: &str,
        location: SourceLocation,
    ) -> Result<usize, MacroError> {
        self.class_registry
            .as_ref()
            .and_then(|registry| registry.enum_constructor_names(owner))
            .and_then(|names| names.iter().position(|candidate| candidate == name))
            .ok_or_else(|| MacroError::RuntimeError {
                message: format!("unknown enum constructor {owner}.{name}"),
                location,
            })
    }

    fn enum_names(&self, owner: &str, location: SourceLocation) -> Result<Vec<String>, MacroError> {
        self.class_registry
            .as_ref()
            .and_then(|registry| registry.enum_constructor_names(owner))
            .ok_or_else(|| MacroError::RuntimeError {
                message: format!("{owner} is not an enum"),
                location,
            })
    }

    fn host_type(
        &mut self,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<Option<MacroValue>, MacroError> {
        let target = args.first().cloned().unwrap_or(MacroValue::Null);
        let class = match &target {
            MacroValue::Class(name) => Some(name.to_string()),
            _ => None,
        };
        let parameters = |index: usize| match args.get(index) {
            Some(MacroValue::Array(items)) => items.read().to_vec(),
            _ => Vec::new(),
        };
        Ok(Some(match (method, class) {
            ("getClassName" | "getEnumName", Some(name)) => MacroValue::from_string(name),
            ("getClass", None) => match &target {
                MacroValue::Object(fields) => fields
                    .read()
                    .get("__type__")
                    .and_then(MacroValue::as_string)
                    .map_or(MacroValue::Null, |name| MacroValue::Class(Arc::from(name))),
                _ => return Ok(None),
            },
            ("getEnum", None) => match &target {
                MacroValue::Enum(owner, _, _) => MacroValue::Class(owner.clone()),
                _ => super::object_enum_constructor(&target)
                    .map_or(MacroValue::Null, |(owner, _)| {
                        MacroValue::Class(Arc::from(owner))
                    }),
            },
            ("resolveClass" | "resolveEnum", None) => {
                let name = string_arg(args, 0, "Type.resolveClass", location)?;
                let registry = self.class_registry.as_ref();
                let is_enum = registry
                    .is_some_and(|registry| registry.enum_constructor_names(name).is_some());
                let found = if method == "resolveEnum" {
                    is_enum
                } else {
                    !is_enum
                        && (MacroInterpreter::intrinsic_class(name)
                            || registry.is_some_and(|registry| registry.find_class(name).is_some()))
                };
                if found {
                    MacroValue::Class(Arc::from(name))
                } else {
                    MacroValue::Null
                }
            }
            ("getSuperClass", Some(name)) => self
                .class_registry
                .as_ref()
                .and_then(|registry| registry.find_class(&name))
                .and_then(|class| class.parent.clone())
                .map_or(MacroValue::Null, |parent| {
                    let parent = self.runtime_class_name(&parent).unwrap_or(parent);
                    MacroValue::Class(Arc::from(parent))
                }),
            ("getInstanceFields" | "getClassFields", Some(name)) => {
                let Some(registry) = self.class_registry.as_ref() else {
                    return Ok(None);
                };
                let mut names = Vec::new();
                let mut current = registry.find_class(&name);
                while let Some(class) = current {
                    if method == "getClassFields" {
                        names.extend(class.static_vars.iter().map(|field| field.name.clone()));
                        names.extend(class.static_methods.keys().cloned());
                        break;
                    }
                    names.extend(class.instance_vars.iter().map(|field| field.name.clone()));
                    names.extend(class.instance_methods.keys().cloned());
                    current = registry.parent_class(class);
                }
                let mut seen = std::collections::BTreeSet::new();
                names.retain(|name| seen.insert(name.clone()));
                MacroValue::array(names.into_iter().map(MacroValue::from_string).collect())
            }
            ("createInstance", Some(name)) => {
                let resolved = self.resolve_class_name(&name);
                self.construct_named(&name, resolved, parameters(1), location)?
            }
            ("createEmptyInstance", Some(name)) => MacroValue::object(BTreeMap::from([(
                "__type__".to_string(),
                MacroValue::from_string(name),
            )])),
            ("getEnumConstructs", Some(name)) => MacroValue::array(
                self.enum_names(&name, location)?
                    .into_iter()
                    .map(MacroValue::from_string)
                    .collect(),
            ),
            ("createEnum", Some(name)) => {
                let constructor = string_arg(args, 1, "Type.createEnum", location)?.to_string();
                self.enum_value(&name, &constructor, parameters(2), location)?
            }
            ("createEnumIndex", Some(name)) => {
                let names = self.enum_names(&name, location)?;
                let constructor = int_arg(args, 1)
                    .and_then(|index| names.get(usize::try_from(index).ok()?))
                    .ok_or_else(|| MacroError::RuntimeError {
                        message: format!("Type.createEnumIndex: no such index in {name}"),
                        location,
                    })?
                    .clone();
                self.enum_value(&name, &constructor, parameters(2), location)?
            }
            ("allEnums", Some(name)) => {
                let mut values = Vec::new();
                for constructor in self.enum_names(&name, location)? {
                    let arity = self
                        .class_registry
                        .as_ref()
                        .and_then(|registry| registry.enum_constructor(Some(&name), &constructor))
                        .map_or(0, |(_, arity)| arity);
                    if arity == 0 {
                        values.push(self.enum_value(&name, &constructor, Vec::new(), location)?);
                    }
                }
                MacroValue::array(values)
            }
            ("enumIndex", None) => match &target {
                MacroValue::Enum(owner, name, _) => {
                    MacroValue::Int(self.enum_index(owner, name, location)? as i64)
                }
                _ => return Ok(None),
            },
            ("typeof", _) => {
                let class = |name: &str| ("TClass", vec![MacroValue::Class(Arc::from(name))]);
                let (constructor, arguments) = match &target {
                    MacroValue::Null => ("TNull", Vec::new()),
                    MacroValue::Int(_) => ("TInt", Vec::new()),
                    MacroValue::Float(_) => ("TFloat", Vec::new()),
                    MacroValue::Bool(_) => ("TBool", Vec::new()),
                    MacroValue::Function(_) => ("TFunction", Vec::new()),
                    MacroValue::String(_) => class("String"),
                    MacroValue::Array(_) => class("Array"),
                    MacroValue::Map(_) => class("haxe.ds.StringMap"),
                    MacroValue::Enum(owner, _, _) => {
                        ("TEnum", vec![MacroValue::Class(owner.clone())])
                    }
                    MacroValue::Object(fields) => match fields
                        .read()
                        .get("__type__")
                        .and_then(MacroValue::as_string)
                    {
                        Some(name) => class(name),
                        None => ("TObject", Vec::new()),
                    },
                    MacroValue::Class(_) => ("TObject", Vec::new()),
                    _ => ("TUnknown", Vec::new()),
                };
                MacroValue::Enum(
                    Arc::from("ValueType"),
                    Arc::from(constructor),
                    Arc::new(arguments),
                )
            }
            _ => return Ok(None),
        }))
    }

    /// `new Date(year, month, day, hour, min, sec)` in local time.
    pub(super) fn new_date(
        &self,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let part = |index: usize| int_arg(args, index).unwrap_or(0);
        let time = chrono::NaiveDate::from_ymd_opt(part(0) as i32, 1, 1)
            .and_then(|date| date.checked_add_months(chrono::Months::new(part(1).max(0) as u32)))
            .and_then(|date| date.checked_add_days(chrono::Days::new((part(2) - 1).max(0) as u64)))
            .and_then(|date| date.and_hms_opt(0, 0, 0))
            .map(|time| time + chrono::Duration::seconds(part(3) * 3600 + part(4) * 60 + part(5)))
            .and_then(local_milliseconds)
            .ok_or_else(|| MacroError::RuntimeError {
                message: "new Date: the fields do not name a local time".into(),
                location,
            })?;
        Ok(date_value(time))
    }

    /// Instance methods of the host-supplied std objects; `None` when `type_name` is not one.
    pub(super) fn host_std_method(
        &mut self,
        object: &Shared<BTreeMap<String, MacroValue>>,
        type_name: &str,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Option<Result<MacroValue, MacroError>> {
        Some(match type_name {
            "Date" => date_method(object, method, location),
            "haxe.io.Bytes" => bytes_method(object, method, args, location),
            "sys.io.FileInput" => file_input_method(object, method, args, location),
            "sys.io.FileOutput" | "sys.io.Stdout" | "sys.io.Stderr" => {
                self.output_method(object, type_name, method, args, location)
            }
            _ => return None,
        })
    }

    fn output_method(
        &mut self,
        object: &Shared<BTreeMap<String, MacroValue>>,
        type_name: &str,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let big_endian = object.read().get("bigEndian") == Some(&MacroValue::Bool(true));
        let number = |bytes: Vec<u8>| -> Vec<u8> {
            if big_endian {
                bytes.into_iter().rev().collect()
            } else {
                bytes
            }
        };
        let int = int_arg(args, 0).unwrap_or(0);
        let float = args.first().and_then(MacroValue::as_float).unwrap_or(0.0);
        let mut result = MacroValue::Null;
        let written: Vec<u8> = match method {
            "writeByte" | "writeInt8" => vec![int as u8],
            "writeInt16" | "writeUInt16" => number((int as u16).to_le_bytes().to_vec()),
            "writeInt24" | "writeUInt24" => number((int as u32).to_le_bytes()[..3].to_vec()),
            "writeInt32" => number((int as u32).to_le_bytes().to_vec()),
            "writeFloat" => number((float as f32).to_le_bytes().to_vec()),
            "writeDouble" => number(float.to_le_bytes().to_vec()),
            "writeString" => string_arg(args, 0, "Output.writeString", location)?
                .as_bytes()
                .to_vec(),
            "write" => bytes_arg(args, 0, "Output.write", location)?,
            "writeBytes" | "writeFullBytes" => {
                let bytes = bytes_arg(args, 0, "Output.writeBytes", location)?;
                let range = checked_range(
                    int_arg(args, 1).unwrap_or(0),
                    int_arg(args, 2).unwrap_or(0),
                    bytes.len(),
                    "Output.writeBytes",
                    location,
                )?;
                if method == "writeBytes" {
                    result = MacroValue::Int(range.len() as i64);
                }
                bytes[range].to_vec()
            }
            "flush" | "close" | "prepare" => Vec::new(),
            _ => {
                return Err(MacroError::UnsupportedOperation {
                    operation: format!("{type_name}.{method}()"),
                    location,
                });
            }
        };
        match type_name {
            "sys.io.Stdout" if !written.is_empty() => {
                self.trace_output
                    .push(String::from_utf8_lossy(&written).into_owned());
            }
            "sys.io.Stderr" if !written.is_empty() => {
                use std::io::Write;
                let _ = std::io::stderr().lock().write_all(&written);
            }
            "sys.io.FileOutput" => {
                let fields = object.read();
                if let Some(MacroValue::Array(buffer)) = fields.get("__buffer__") {
                    buffer.write(|buffer| {
                        buffer.extend(written.iter().map(|byte| MacroValue::Int((*byte).into())))
                    });
                    if matches!(method, "flush" | "close") {
                        let pending: Vec<u8> = buffer
                            .read()
                            .iter()
                            .map(|byte| byte.as_int().unwrap_or(0) as u8)
                            .collect();
                        let path = fields
                            .get("__path__")
                            .and_then(MacroValue::as_string)
                            .unwrap_or("");
                        use std::io::Write;
                        std::fs::OpenOptions::new()
                            .append(true)
                            .open(path)
                            .and_then(|mut file| file.write_all(&pending))
                            .map_err(|error| MacroError::RuntimeError {
                                message: format!("FileOutput.{method}('{path}') failed: {error}"),
                                location,
                            })?;
                        buffer.write(Vec::clear);
                        if let Some(dependencies) = self.dependencies() {
                            dependencies.refresh_written(path);
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(result)
    }
}

/// Process CPU time in seconds.
fn cpu_time() -> Option<f64> {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: getrusage fills the struct it is given and reads nothing else.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
            return None;
        }
        // SAFETY: a successful getrusage initialised every field.
        let usage = unsafe { usage.assume_init() };
        let seconds = |time: libc::timeval| time.tv_sec as f64 + time.tv_usec as f64 / 1e6;
        Some(seconds(usage.ru_utime) + seconds(usage.ru_stime))
    }
    #[cfg(not(unix))]
    {
        None
    }
}

fn date_method(
    object: &Shared<BTreeMap<String, MacroValue>>,
    method: &str,
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    let milliseconds = object
        .read()
        .get("__time__")
        .and_then(MacroValue::as_float)
        .unwrap_or(0.0);
    if method == "getTime" {
        return Ok(MacroValue::Float(milliseconds));
    }
    if method == "toString" {
        return Ok(MacroValue::from_string(date_string(milliseconds)));
    }
    let invalid = || MacroError::RuntimeError {
        message: format!("Date.{method}: time {milliseconds} is out of range"),
        location,
    };
    let local = local_time(milliseconds).ok_or_else(invalid)?;
    let utc = local.with_timezone(&chrono::Utc);
    let value = match method {
        "getFullYear" => i64::from(local.year()),
        "getMonth" => i64::from(local.month0()),
        "getDate" => i64::from(local.day()),
        "getDay" => i64::from(local.weekday().num_days_from_sunday()),
        "getHours" => i64::from(local.hour()),
        "getMinutes" => i64::from(local.minute()),
        "getSeconds" => i64::from(local.second()),
        "getUTCFullYear" => i64::from(utc.year()),
        "getUTCMonth" => i64::from(utc.month0()),
        "getUTCDate" => i64::from(utc.day()),
        "getUTCDay" => i64::from(utc.weekday().num_days_from_sunday()),
        "getUTCHours" => i64::from(utc.hour()),
        "getUTCMinutes" => i64::from(utc.minute()),
        "getUTCSeconds" => i64::from(utc.second()),
        "getTimezoneOffset" => -i64::from(local.offset().local_minus_utc()) / 60,
        _ => {
            return Err(MacroError::UnsupportedOperation {
                operation: format!("Date.{method}()"),
                location,
            });
        }
    };
    Ok(MacroValue::Int(value))
}

fn bytes_method(
    object: &Shared<BTreeMap<String, MacroValue>>,
    method: &str,
    args: &[MacroValue],
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    let this = MacroValue::Object(object.clone());
    let storage = bytes_storage(&this).ok_or_else(|| MacroError::TypeError {
        message: "invalid Bytes storage".into(),
        location,
    })?;
    let length = storage.read().len();
    let operation = format!("Bytes.{method}");
    let pos = int_arg(args, 0).unwrap_or(0);
    let range = |len: i64| checked_range(pos, len, length, &operation, location);
    let read = |len: i64| -> Result<Vec<u8>, MacroError> {
        let range = range(len)?;
        let data = storage.read();
        Ok(range.map(|index| byte_at(&data, index)).collect())
    };
    let store = |bytes: &[u8]| -> Result<(), MacroError> {
        let range = range(bytes.len() as i64)?;
        storage.write(|data| {
            for (slot, byte) in data[range].iter_mut().zip(bytes) {
                *slot = MacroValue::Int((*byte).into());
            }
        });
        Ok(())
    };
    let value = args.get(1);
    let int = value.and_then(MacroValue::as_int).unwrap_or(0);
    let float = value.and_then(MacroValue::as_float).unwrap_or(0.0);
    Ok(match method {
        "get" => MacroValue::Int(read(1)?[0].into()),
        "set" => {
            store(&[int as u8])?;
            MacroValue::Null
        }
        "getUInt16" => MacroValue::Int(u16::from_le_bytes(read(2)?.try_into().unwrap()).into()),
        "getInt32" => MacroValue::Int(i32::from_le_bytes(read(4)?.try_into().unwrap()).into()),
        "getInt64" => MacroValue::Int(i64::from_le_bytes(read(8)?.try_into().unwrap())),
        "getFloat" => MacroValue::Float(f32::from_le_bytes(read(4)?.try_into().unwrap()).into()),
        "getDouble" => MacroValue::Float(f64::from_le_bytes(read(8)?.try_into().unwrap())),
        "setUInt16" => {
            store(&(int as u16).to_le_bytes())?;
            MacroValue::Null
        }
        "setInt32" => {
            store(&(int as u32).to_le_bytes())?;
            MacroValue::Null
        }
        "setInt64" => {
            store(&int.to_le_bytes())?;
            MacroValue::Null
        }
        "setFloat" => {
            store(&(float as f32).to_le_bytes())?;
            MacroValue::Null
        }
        "setDouble" => {
            store(&float.to_le_bytes())?;
            MacroValue::Null
        }
        "sub" => super::super::assets::bytes_value(read(int)?),
        "getString" | "readString" => {
            MacroValue::from_string(String::from_utf8_lossy(&read(int)?).into_owned())
        }
        "fill" => {
            let byte = int_arg(args, 2).unwrap_or(0) as u8;
            store(&vec![byte; usize::try_from(int).unwrap_or(0)])?;
            MacroValue::Null
        }
        "blit" => {
            let source = bytes_arg(args, 1, &operation, location)?;
            let source_range = checked_range(
                int_arg(args, 2).unwrap_or(0),
                int_arg(args, 3).unwrap_or(0),
                source.len(),
                &operation,
                location,
            )?;
            store(&source[source_range])?;
            MacroValue::Null
        }
        "compare" => {
            let other = bytes_arg(args, 0, &operation, location)?;
            let data: Vec<u8> = (0..length)
                .map(|index| byte_at(&storage.read(), index))
                .collect();
            MacroValue::Int(match data.cmp(&other) {
                std::cmp::Ordering::Less => -1,
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Greater => 1,
            })
        }
        "toHex" => MacroValue::from_string(hex(&read(length as i64 - pos)?)),
        "toString" => MacroValue::from_string(
            String::from_utf8_lossy(&read(length as i64 - pos)?).into_owned(),
        ),
        _ => {
            return Err(MacroError::UnsupportedOperation {
                operation: format!("{operation}()"),
                location,
            });
        }
    })
}

fn file_input_method(
    object: &Shared<BTreeMap<String, MacroValue>>,
    method: &str,
    args: &[MacroValue],
    location: SourceLocation,
) -> Result<MacroValue, MacroError> {
    let fields = object.read();
    let Some(MacroValue::Array(storage)) = fields.get("__bytes__") else {
        return Err(MacroError::TypeError {
            message: "invalid FileInput".into(),
            location,
        });
    };
    let data = storage.read();
    let pos = fields
        .get("__pos__")
        .and_then(MacroValue::as_int)
        .unwrap_or(0)
        .clamp(0, data.len() as i64) as usize;
    let big_endian = fields.get("bigEndian") == Some(&MacroValue::Bool(true));
    drop(fields);
    let remaining = data.len() - pos;
    let take = |count: usize| -> Result<Vec<u8>, MacroError> {
        if count > remaining {
            return Err(eof(location));
        }
        Ok((pos..pos + count)
            .map(|index| byte_at(&data, index))
            .collect())
    };
    let number = |count: usize| -> Result<Vec<u8>, MacroError> {
        let mut bytes = take(count)?;
        if big_endian {
            bytes.reverse();
        }
        Ok(bytes)
    };
    let (value, consumed) = match method {
        "readByte" | "readInt8" => {
            let byte = take(1)?[0];
            let value = if method == "readInt8" {
                i64::from(byte as i8)
            } else {
                byte.into()
            };
            (MacroValue::Int(value), 1)
        }
        "readInt16" => (
            MacroValue::Int(i16::from_le_bytes(number(2)?.try_into().unwrap()).into()),
            2,
        ),
        "readUInt16" => (
            MacroValue::Int(u16::from_le_bytes(number(2)?.try_into().unwrap()).into()),
            2,
        ),
        "readInt32" => (
            MacroValue::Int(i32::from_le_bytes(number(4)?.try_into().unwrap()).into()),
            4,
        ),
        "readFloat" => (
            MacroValue::Float(f32::from_le_bytes(number(4)?.try_into().unwrap()).into()),
            4,
        ),
        "readDouble" => (
            MacroValue::Float(f64::from_le_bytes(number(8)?.try_into().unwrap())),
            8,
        ),
        "readString" => {
            let count = usize::try_from(int_arg(args, 0).unwrap_or(0)).unwrap_or(0);
            (
                MacroValue::from_string(String::from_utf8_lossy(&take(count)?).into_owned()),
                count,
            )
        }
        "read" => {
            let count = usize::try_from(int_arg(args, 0).unwrap_or(0)).unwrap_or(0);
            (super::super::assets::bytes_value(take(count)?), count)
        }
        "readAll" => (
            super::super::assets::bytes_value(take(remaining)?),
            remaining,
        ),
        "readLine" | "readUntil" => {
            if remaining == 0 {
                return Err(eof(location));
            }
            let end = if method == "readLine" {
                b'\n'
            } else {
                int_arg(args, 0).unwrap_or(0) as u8
            };
            let found = (pos..data.len()).find(|&index| byte_at(&data, index) == end);
            let stop = found.unwrap_or(data.len());
            let mut line: Vec<u8> = (pos..stop).map(|index| byte_at(&data, index)).collect();
            let consumed = stop - pos + usize::from(found.is_some());
            if method == "readLine" && line.last() == Some(&b'\r') {
                line.pop();
            }
            (
                MacroValue::from_string(String::from_utf8_lossy(&line).into_owned()),
                consumed,
            )
        }
        "readBytes" | "readFullBytes" => {
            let target =
                args.first()
                    .and_then(bytes_storage)
                    .ok_or_else(|| MacroError::TypeError {
                        message: format!("FileInput.{method} expects Bytes"),
                        location,
                    })?;
            let offset = int_arg(args, 1).unwrap_or(0);
            let wanted = int_arg(args, 2).unwrap_or(0);
            let range = checked_range(
                offset,
                wanted,
                target.read().len(),
                &format!("FileInput.{method}"),
                location,
            )?;
            let count = if method == "readBytes" {
                if remaining == 0 && !range.is_empty() {
                    return Err(eof(location));
                }
                range.len().min(remaining)
            } else {
                range.len()
            };
            let bytes = take(count)?;
            target.write(|target| {
                for (slot, byte) in target[range.start..range.start + count]
                    .iter_mut()
                    .zip(&bytes)
                {
                    *slot = MacroValue::Int((*byte).into());
                }
            });
            let result = if method == "readBytes" {
                MacroValue::Int(count as i64)
            } else {
                MacroValue::Null
            };
            (result, count)
        }
        "eof" => (MacroValue::Bool(remaining == 0), 0),
        "tell" => (MacroValue::Int(pos as i64), 0),
        "close" => (MacroValue::Null, 0),
        "seek" => {
            let offset = int_arg(args, 0).unwrap_or(0);
            let base = match args.get(1) {
                Some(MacroValue::Enum(_, origin, _)) if &**origin == "SeekCur" => pos as i64,
                Some(MacroValue::Enum(_, origin, _)) if &**origin == "SeekEnd" => data.len() as i64,
                _ => 0,
            };
            let target = (base + offset).clamp(0, data.len() as i64);
            drop(data);
            object.write(|fields| fields.insert("__pos__".into(), MacroValue::Int(target)));
            return Ok(MacroValue::Null);
        }
        _ => {
            return Err(MacroError::UnsupportedOperation {
                operation: format!("FileInput.{method}()"),
                location,
            });
        }
    };
    drop(data);
    if consumed > 0 {
        let next = MacroValue::Int((pos + consumed) as i64);
        object.write(|fields| fields.insert("__pos__".into(), next));
    }
    Ok(value)
}
