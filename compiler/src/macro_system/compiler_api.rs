//! Effects of initialization macros on subsequent module typing.

use super::{MacroError, MacroValue};
use crate::tast::SourceLocation;

#[derive(Clone)]
pub struct GlobalMetadata {
    pub path: String,
    pub metadata: parser::Metadata,
    pub recursive: bool,
    pub to_types: bool,
    pub to_fields: bool,
}

impl GlobalMetadata {
    pub fn from_args(args: &[MacroValue], location: SourceLocation) -> Result<Self, MacroError> {
        let fail = |message: String| MacroError::ContextError {
            method: "Compiler.addGlobalMetadata".into(),
            message,
            location,
        };
        let (Some(MacroValue::String(path)), Some(MacroValue::String(text))) =
            (args.first(), args.get(1))
        else {
            return Err(fail("expected a path and metadata string".into()));
        };
        let metadata = parser::rd::rd_parse_metadata(text).map_err(|e| fail(e.message))?;
        Ok(Self {
            path: path.to_string(),
            metadata,
            recursive: args.get(2).is_none_or(MacroValue::is_truthy),
            to_types: args.get(3).is_none_or(MacroValue::is_truthy),
            to_fields: args.get(4).is_some_and(MacroValue::is_truthy),
        })
    }

    fn matches(&self, path: &str) -> bool {
        if self.path.is_empty() {
            self.recursive || !path.contains('.')
        } else if self.recursive {
            path.starts_with(&self.path)
        } else {
            path == self.path
        }
    }

    pub fn apply(&self, file: &mut parser::HaxeFile) {
        let pack = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();
        for decl in &mut file.declarations {
            let span = decl.span();
            let (name, meta, fields) = match decl {
                parser::TypeDeclaration::Class(c) => (&c.name, &mut c.meta, Some(&mut c.fields)),
                parser::TypeDeclaration::Interface(c) => {
                    (&c.name, &mut c.meta, Some(&mut c.fields))
                }
                parser::TypeDeclaration::Abstract(c) => (&c.name, &mut c.meta, Some(&mut c.fields)),
                parser::TypeDeclaration::Enum(c) => (&c.name, &mut c.meta, None),
                parser::TypeDeclaration::Typedef(c) => (&c.name, &mut c.meta, None),
                parser::TypeDeclaration::Conditional(_) => continue,
            };
            let path = if pack.is_empty() {
                name.clone()
            } else {
                format!("{pack}.{name}")
            };
            if self.to_types && self.matches(&path) {
                let mut metadata = self.metadata.clone();
                metadata.span = span;
                meta.push(metadata);
            }
            if self.to_fields
                && let Some(fields) = fields
            {
                for field in fields {
                    let name = match &field.kind {
                        parser::ClassFieldKind::Function(f) => &f.name,
                        parser::ClassFieldKind::Var { name, .. }
                        | parser::ClassFieldKind::Final { name, .. }
                        | parser::ClassFieldKind::Property { name, .. } => name,
                    };
                    if self.matches(&format!("{path}.{name}")) {
                        field.meta.push(self.metadata.clone());
                    }
                }
            }
        }
    }
}
