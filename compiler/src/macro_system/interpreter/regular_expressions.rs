use super::*;

impl MacroInterpreter {
    pub(super) fn new_regexp(
        &self,
        pattern: &str,
        flags: &str,
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        compile(pattern, flags, location)?;
        Ok(MacroValue::Object(Arc::new(BTreeMap::from([
            ("__type__".into(), MacroValue::from_str("EReg")),
            ("pattern".into(), MacroValue::from_str(pattern)),
            ("flags".into(), MacroValue::from_str(flags)),
        ]))))
    }

    pub(super) fn regexp_method(
        &mut self,
        object: &Arc<BTreeMap<String, MacroValue>>,
        method: &str,
        args: &[MacroValue],
        location: SourceLocation,
    ) -> Result<MacroValue, MacroError> {
        let pattern = object
            .get("pattern")
            .and_then(MacroValue::as_string)
            .unwrap_or("");
        let flags = object
            .get("flags")
            .and_then(MacroValue::as_string)
            .unwrap_or("");
        let regex = compile(pattern, flags, location)?;
        let input = args.first().and_then(MacroValue::as_string).unwrap_or("");
        let text = |key: &str| {
            object
                .get(key)
                .and_then(MacroValue::as_string)
                .map(str::to_string)
                .ok_or_else(|| MacroError::TypeError {
                    message: "regular expression has no match".into(),
                    location,
                })
        };
        match method {
            "match" | "matchSub" => {
                let offset = if method == "matchSub" {
                    args.get(1).and_then(MacroValue::as_int).unwrap_or(0).max(0) as usize
                } else {
                    0
                };
                let end = args
                    .get(2)
                    .and_then(MacroValue::as_int)
                    .filter(|length| *length >= 0)
                    .map(|length| offset.saturating_add(length as usize))
                    .unwrap_or(input.len())
                    .min(input.len());
                let matched = input
                    .get(offset..end)
                    .and_then(|input| regex.captures(input));
                let found = matched.is_some();
                let mut updated = (**object).clone();
                updated.remove("groups");
                updated.remove("left");
                updated.remove("right");
                updated.remove("position");
                if let Some(captures) = matched {
                    let whole = captures.get(0).unwrap();
                    updated.insert(
                        "groups".into(),
                        MacroValue::Array(Arc::new(
                            captures
                                .iter()
                                .map(|capture| {
                                    capture.map_or(MacroValue::Null, |capture| {
                                        MacroValue::from_str(capture.as_str())
                                    })
                                })
                                .collect(),
                        )),
                    );
                    updated.insert(
                        "left".into(),
                        MacroValue::from_str(&input[..offset + whole.start()]),
                    );
                    updated.insert(
                        "right".into(),
                        MacroValue::from_str(&input[offset + whole.end()..]),
                    );
                    updated.insert(
                        "position".into(),
                        MacroValue::Object(Arc::new(BTreeMap::from([
                            (
                                "pos".into(),
                                MacroValue::Int((offset + whole.start()) as i64),
                            ),
                            ("len".into(), MacroValue::Int(whole.len() as i64)),
                        ]))),
                    );
                }
                self.replace_references(
                    &MacroValue::Object(object.clone()),
                    &MacroValue::Object(Arc::new(updated)),
                );
                Ok(MacroValue::Bool(found))
            }
            "matched" => object
                .get("groups")
                .and_then(MacroValue::as_array)
                .and_then(|groups| {
                    groups.get(args.first().and_then(MacroValue::as_int).unwrap_or(0) as usize)
                })
                .cloned()
                .ok_or_else(|| MacroError::TypeError {
                    message: "regular expression has no such group".into(),
                    location,
                }),
            "matchedLeft" => text("left").map(MacroValue::from_string),
            "matchedRight" => text("right").map(MacroValue::from_string),
            "matchedPos" => object
                .get("position")
                .cloned()
                .ok_or_else(|| MacroError::TypeError {
                    message: "regular expression has no match".into(),
                    location,
                }),
            "replace" => {
                let replacement = args.get(1).and_then(MacroValue::as_string).unwrap_or("");
                let result = if flags.contains('g') {
                    regex.replace_all(input, replacement)
                } else {
                    regex.replace(input, replacement)
                };
                Ok(MacroValue::from_str(&result))
            }
            "split" => Ok(MacroValue::Array(Arc::new(
                regex.split(input).map(MacroValue::from_str).collect(),
            ))),
            _ => Err(MacroError::UnsupportedOperation {
                operation: format!("EReg.{method}"),
                location,
            }),
        }
    }
}

fn compile(
    pattern: &str,
    flags: &str,
    location: SourceLocation,
) -> Result<regex::Regex, MacroError> {
    regex::RegexBuilder::new(pattern)
        .case_insensitive(flags.contains('i'))
        .multi_line(flags.contains('m'))
        .dot_matches_new_line(flags.contains('s'))
        .ignore_whitespace(flags.contains('x'))
        .build()
        .map_err(|error| MacroError::TypeError {
            message: format!("invalid regular expression: {error}"),
            location,
        })
}
