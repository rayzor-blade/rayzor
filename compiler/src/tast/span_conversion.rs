//! Utilities for converting parser spans to TAST source locations
//!
//! This module provides conversion between the parser's Span type and
//! the TAST's SourceLocation type, using the unified source mapping system.

use crate::tast::symbols::SourceLocation;
use parser::Span;
use source_map::{
    parser_integration::{ParserSpan, SpanConversion as SourceMapSpanConversion},
    FileId, SourceMap,
};

/// Bridge between source_map types and TAST types
pub struct SpanConverter {
    /// Source map for all files (private to this converter; used for
    /// byte-offset → line/column resolution against the file's bytes).
    source_map: SourceMap,
    /// Current file in the PRIVATE source_map. Always FileId(0) for
    /// `with_file`-constructed converters; used only for line/column
    /// lookup, NOT for the file_id emitted on SourceLocations.
    current_file_id: FileId,
    /// External file_id supplied by the outer compilation pipeline.
    /// THIS is the file_id stamped on every SourceLocation produced
    /// by `convert_span`, so per-file lowerings end up with spans
    /// that correctly identify the imported module they came from
    /// (rather than every span carrying file_id=0 because the
    /// private source_map starts fresh per converter).
    external_file_id: u32,
}

impl SpanConverter {
    /// Create a new span converter with an existing source map and current file.
    /// `external_file_id` defaults to the internal current_file_id's raw value.
    pub fn new(source_map: SourceMap, current_file_id: FileId) -> Self {
        Self {
            source_map,
            current_file_id,
            external_file_id: current_file_id.as_usize() as u32,
        }
    }

    /// Create a span converter with a new file and a specific external file_id.
    ///
    /// The internal source_map gets file_id 0 (the file is the first one
    /// added), but every SourceLocation emitted by `convert_span` carries
    /// `external_file_id` instead — this is how cross-file lowerings end
    /// up tagging their TypedExpression spans with the real compilation-
    /// level file_id, not the always-0 of the private source_map.
    pub fn with_file_and_id(file_name: String, source_text: String, external_file_id: u32) -> Self {
        let mut source_map = SourceMap::new();
        let file_id = source_map.add_file(file_name, source_text);
        Self {
            source_map,
            current_file_id: file_id,
            external_file_id,
        }
    }

    /// Create a span converter with a new file (preserved for compatibility;
    /// `external_file_id` is taken from the internal source_map and so will
    /// always be 0 — prefer `with_file_and_id` from new call sites).
    pub fn with_file(file_name: String, source_text: String) -> Self {
        Self::with_file_and_id(file_name, source_text, 0)
    }

    /// Add a new file to the source map and return its FileId
    pub fn add_file(&mut self, file_name: String, source_text: String) -> FileId {
        self.source_map.add_file(file_name, source_text)
    }

    /// Set the current file being processed.
    /// Also resets `external_file_id` to the raw value of the new
    /// FileId so subsequent `convert_span` calls tag SourceLocations
    /// with the matching file_id. Callers that need to keep an
    /// independent external_file_id can call `set_external_file_id`
    /// after this.
    pub fn set_current_file(&mut self, file_id: FileId) {
        self.current_file_id = file_id;
        self.external_file_id = file_id.as_usize() as u32;
    }

    /// Override just the external_file_id used when emitting
    /// SourceLocations (without touching the internal source_map
    /// lookup id). Used by callers that share a converter across
    /// files of different compilation-level identities.
    pub fn set_external_file_id(&mut self, external_file_id: u32) {
        self.external_file_id = external_file_id;
    }

    /// Convert a parser span to a TAST source location.
    ///
    /// `file_id` on the returned SourceLocation is `external_file_id`
    /// (the compilation-pipeline-level id), NOT the private source_map's
    /// internal id (which is always 0 for `with_file`-style converters).
    pub fn convert_span(&self, span: Span) -> SourceLocation {
        let parser_span = ParserSpan::new(span.start, span.end);
        if let Some(source_span) =
            parser_span.to_source_span(self.current_file_id, &self.source_map)
        {
            SourceLocation {
                file_id: self.external_file_id,
                line: source_span.start.line as u32,
                column: source_span.start.column as u32,
                byte_offset: source_span.start.byte_offset as u32,
            }
        } else {
            SourceLocation::unknown()
        }
    }

    /// Create a source location for unknown/synthetic nodes
    pub fn unknown_location(&self) -> SourceLocation {
        SourceLocation::unknown()
    }

    /// Merge two spans and convert to source location
    pub fn merge_spans(&self, span1: Span, span2: Span) -> SourceLocation {
        let merged = span1.merge(span2);
        self.convert_span(merged)
    }

    /// Get access to the underlying source map
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    /// Get mutable access to the underlying source map
    pub fn source_map_mut(&mut self) -> &mut SourceMap {
        &mut self.source_map
    }
}

/// Extension trait for optional span conversion using the new system
pub trait SpanConversion {
    fn to_source_location(&self, converter: &SpanConverter) -> SourceLocation;
}

impl SpanConversion for Span {
    fn to_source_location(&self, converter: &SpanConverter) -> SourceLocation {
        converter.convert_span(*self)
    }
}

impl SpanConversion for Option<Span> {
    fn to_source_location(&self, converter: &SpanConverter) -> SourceLocation {
        match self {
            Some(span) => converter.convert_span(*span),
            None => converter.unknown_location(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_span_conversion() {
        let source = "line1\nline2\nline3";
        let converter = SpanConverter::with_file("test.hx".to_string(), source.to_string());

        // Test first line
        let span1 = Span::new(0, 5);
        let loc1 = converter.convert_span(span1);
        assert_eq!(loc1.line, 1);
        assert_eq!(loc1.column, 1);
        assert_eq!(loc1.byte_offset, 0);

        // Test second line
        let span2 = Span::new(6, 11);
        let loc2 = converter.convert_span(span2);
        assert_eq!(loc2.line, 2);
        assert_eq!(loc2.column, 1);
        assert_eq!(loc2.byte_offset, 6);

        // Test middle of third line
        let span3 = Span::new(14, 16);
        let loc3 = converter.convert_span(span3);
        assert_eq!(loc3.line, 3);
        assert_eq!(loc3.column, 3);
        assert_eq!(loc3.byte_offset, 14);
    }

    #[test]
    fn test_multi_file_support() {
        let mut converter =
            SpanConverter::with_file("file1.hx".to_string(), "hello\nworld".to_string());
        let file2_id = converter.add_file("file2.hx".to_string(), "foo\nbar\nbaz".to_string());

        // Test first file
        let span1 = Span::new(0, 5);
        let loc1 = converter.convert_span(span1);
        assert_eq!(loc1.line, 1);
        assert_eq!(loc1.column, 1);

        // Switch to second file
        converter.set_current_file(file2_id);
        let span2 = Span::new(4, 7); // "bar"
        let loc2 = converter.convert_span(span2);
        assert_eq!(loc2.line, 2);
        assert_eq!(loc2.column, 1);
        assert_eq!(loc2.file_id, file2_id.as_usize() as u32);
    }

    #[test]
    fn test_merge_spans_unified() {
        let converter = SpanConverter::with_file("test.hx".to_string(), "test source".to_string());

        let span1 = Span::new(0, 4);
        let span2 = Span::new(5, 11);

        let merged_loc = converter.merge_spans(span1, span2);
        assert_eq!(merged_loc.byte_offset, 0);
        // The merged span covers from 0 to 11
    }
}
