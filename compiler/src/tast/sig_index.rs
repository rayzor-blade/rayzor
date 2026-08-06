//! Program-wide index of *declared* method signatures, keyed by class.
//!
//! Files TAST-lower in dependency/convergence order, so a call site can
//! reference `C.m(...)` (or `c.m(...)`) before `C`'s file has lowered — the
//! pre-registered method symbol exists but carries no type, and the call
//! result decays to an untyped Dynamic chain (whose member dispatch then
//! guesses). This index retains every parsed class/abstract's static AND
//! instance signatures (raw AST `Type` hints) so the lowering can type such a
//! symbol from its declaration on demand — fully general, no per-method pins.
//!
//! Populated in `CompilationUnit::parse_file` (the canonical, preprocessed
//! parse), and lazily by `resolve` parsing a not-yet-seen declaring file via
//! the same preprocessor config.

use parser::preprocessor::PreprocessorConfig;
use std::collections::{BTreeMap, BTreeSet};

/// Declared signature of one method (raw AST type hints; `None` where the
/// declaration omitted an annotation).
#[derive(Debug, Clone)]
pub struct StaticMethodSig {
    pub params: Vec<Option<parser::Type>>,
    pub return_type: Option<parser::Type>,
}

#[derive(Debug, Default)]
struct ClassSigs {
    statics: BTreeMap<String, StaticMethodSig>,
    instances: BTreeMap<String, StaticMethodSig>,
    /// Declared non-static `var`/`final`/`property` count — the object's
    /// OWN field-slot count. Inherited slots are added by walking `extends`;
    /// see `instance_field_count`.
    instance_field_count: usize,
    /// Parent class name from `extends`, exactly as written. Needed because a
    /// subclass's object layout is parent slots followed by own slots.
    extends: Option<String>,
    /// Package that declared this class, used to qualify a bare `extends`.
    package: String,
}

impl ClassSigs {
    fn table(&self, is_static: bool) -> &BTreeMap<String, StaticMethodSig> {
        if is_static {
            &self.statics
        } else {
            &self.instances
        }
    }
}

#[derive(Debug, Default)]
pub struct StaticSigIndex {
    /// Preprocessor defines matching `CompilationUnit::parse_file`, so
    /// on-demand parses see the same `#if` branches.
    pub preprocessor: PreprocessorConfig,
    /// Qualified class name → its static signatures.
    classes: BTreeMap<String, ClassSigs>,
    /// Bare class name → all qualified names that declare it.
    bare_to_qualified: BTreeMap<String, BTreeSet<String>>,
    /// Typedef alias (`haxe.io.Bytes`) → target path (`rayzor.Bytes`).
    aliases: BTreeMap<String, String>,
    /// Files already indexed (canonical path string) — parse each once.
    indexed_files: BTreeSet<String>,
    /// Qualified names whose on-demand parse failed — don't retry.
    parse_misses: BTreeSet<String>,
}

impl StaticSigIndex {
    /// Record all class/abstract static signatures and typedef aliases in a
    /// parsed file. Idempotent per file.
    pub fn index_file(&mut self, file: &parser::HaxeFile) {
        if !self.indexed_files.insert(file.filename.clone()) {
            return;
        }
        let package = file
            .package
            .as_ref()
            .map(|p| p.path.join("."))
            .unwrap_or_default();
        for decl in &file.declarations {
            self.index_declaration(&package, decl);
        }
    }

    fn index_declaration(&mut self, package: &str, decl: &parser::TypeDeclaration) {
        use parser::TypeDeclaration;
        match decl {
            TypeDeclaration::Class(c) => {
                let parent = c.extends.as_ref().and_then(Self::type_path_name);
                self.index_fields(package, &c.name, &c.fields, parent);
            }
            TypeDeclaration::Abstract(a) => {
                self.index_fields(package, &a.name, &a.fields, None);
            }
            TypeDeclaration::Typedef(t) => {
                // Only straight `typedef A = pkg.B` renames participate in
                // class lookup (e.g. `haxe.io.Bytes = rayzor.Bytes`).
                if let parser::Type::Path { path, .. } = &t.type_def {
                    let alias = Self::qualify(package, &t.name);
                    // An unqualified target names a type in the SAME package
                    // (`private typedef __Int64 = ___Int64` inside `haxe`).
                    // Classes are indexed under qualified names, so a bare
                    // target would never match one.
                    let target = if path.package.is_empty() {
                        Self::qualify(package, &path.name)
                    } else {
                        format!("{}.{}", path.package.join("."), path.name)
                    };
                    if alias != target {
                        self.aliases.insert(alias, target);
                    }
                }
            }
            // `#if` blocks are expanded by the preprocessor before parsing;
            // a residual Conditional matches lowering's skip.
            _ => {}
        }
    }

    /// Qualified name written in a `Type::Path` (`pk.Base` / `Base`).
    fn type_path_name(ty: &parser::Type) -> Option<String> {
        let parser::Type::Path { path, .. } = ty else {
            return None;
        };
        let mut name = path.package.join(".");
        if !name.is_empty() {
            name.push('.');
        }
        name.push_str(&path.name);
        Some(name)
    }

    fn index_fields(
        &mut self,
        package: &str,
        class_name: &str,
        fields: &[parser::ClassField],
        extends: Option<String>,
    ) {
        let qname = Self::qualify(package, class_name);
        let fresh = !self.classes.contains_key(&qname);
        let entry = self.classes.entry(qname.clone()).or_default();
        if fresh {
            entry.extends = extends;
            entry.package = package.to_string();
        }
        for field in fields {
            let is_static = field
                .modifiers
                .iter()
                .any(|m| matches!(m, parser::Modifier::Static));
            let parser::ClassFieldKind::Function(func) = &field.kind else {
                // Data field: var/final/property. First declaration wins
                // (mirrors the or_insert method tables).
                if !is_static && fresh {
                    entry.instance_field_count += 1;
                }
                continue;
            };
            if func.name == "new" {
                continue;
            }
            let table = if is_static {
                &mut entry.statics
            } else {
                &mut entry.instances
            };
            table
                .entry(func.name.clone())
                .or_insert_with(|| StaticMethodSig {
                    params: func.params.iter().map(|p| p.type_hint.clone()).collect(),
                    return_type: func.return_type.clone(),
                });
        }
        self.bare_to_qualified
            .entry(class_name.to_string())
            .or_default()
            .insert(qname);
    }

    fn qualify(package: &str, name: &str) -> String {
        if package.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", package, name)
        }
    }

    /// Find the declared signature of `class_name::method`. `class_name` may
    /// be qualified or bare. Follows typedef aliases. A bare name resolves
    /// only when exactly ONE indexed class both carries that bare name and
    /// declares the method — never a guess among candidates.
    ///
    /// `resolve_file` maps a qualified name to its source path so a
    /// not-yet-indexed declaring file can be parsed on demand with the same
    /// preprocessor config as the main pipeline.
    pub fn resolve(
        &mut self,
        class_name: &str,
        method: &str,
        is_static: bool,
        resolve_file: &dyn Fn(&str) -> Option<std::path::PathBuf>,
    ) -> Option<StaticMethodSig> {
        // Direct name, following typedef aliases (`haxe.io.Bytes → rayzor.Bytes`).
        let mut name = class_name.to_string();
        for _ in 0..4 {
            self.ensure_indexed(&name, resolve_file);
            if let Some(sig) = self
                .classes
                .get(&name)
                .and_then(|c| c.table(is_static).get(method))
            {
                return Some(sig.clone());
            }
            match self.aliases.get(&name) {
                Some(target) => name = target.clone(),
                None => break,
            }
        }

        // Bare-name resolution: import-created / phantom class symbols often
        // carry no package (`Bytes`, `Int64`). Index the standard homes such
        // a bare stdlib name may live under (the same prefix set the
        // on-demand import loader tries) plus any typedef alias whose bare
        // tail matches, then require a UNIQUE declaring class.
        if !class_name.contains('.') {
            const BARE_STDLIB_PREFIXES: &[&str] = &[
                "haxe",
                "haxe.io",
                "haxe.ds",
                "haxe.iterators",
                "haxe.exceptions",
                "sys",
                "sys.thread",
                "sys.net",
                "rayzor",
                "rayzor.ds",
                "rayzor.concurrent",
            ];
            for prefix in BARE_STDLIB_PREFIXES {
                self.ensure_indexed(&format!("{}.{}", prefix, class_name), resolve_file);
            }
            let alias_targets: Vec<String> = self
                .aliases
                .iter()
                .filter(|(alias, _)| alias.rsplit('.').next() == Some(class_name))
                .map(|(_, target)| target.clone())
                .collect();
            for target in alias_targets {
                self.ensure_indexed(&target, resolve_file);
            }

            let candidates: Vec<&String> = self
                .bare_to_qualified
                .get(class_name)
                .map(|set| {
                    set.iter()
                        .filter(|q| {
                            self.classes
                                .get(*q)
                                .is_some_and(|c| c.table(is_static).contains_key(method))
                        })
                        .collect()
                })
                .unwrap_or_default();
            if let [only] = candidates.as_slice() {
                return self.classes[*only].table(is_static).get(method).cloned();
            }
        }
        None
    }

    /// Declared instance-field count for `class_name` (qualified or unique
    /// bare name; follows typedef aliases; parses the declaring file on
    /// demand). Used for object allocation when a `new` lowers before the
    /// class does.
    /// Total instance slots for `class_name`: its own declared fields plus
    /// every ancestor's, since a subclass object is parent slots followed by
    /// own slots. Returns None only when the class itself is unknown; an
    /// unresolvable ANCESTOR yields None too, because a partial count would
    /// under-allocate the object and silently corrupt the heap.
    pub fn instance_field_count(
        &mut self,
        class_name: &str,
        resolve_file: &dyn Fn(&str) -> Option<std::path::PathBuf>,
    ) -> Option<usize> {
        let mut total = 0usize;
        let mut next = Some(class_name.to_string());
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(name) = next {
            if !seen.insert(name.clone()) {
                break; // cyclic `extends` — stop rather than spin
            }
            let (own, parent, pkg) = self.resolve_class_entry(&name, resolve_file)?;
            total += own;
            // A bare `extends Base` names a type in the subclass's own
            // package first; fall back to the name as written so an imported
            // parent still resolves through the unambiguous bare form.
            next = parent.map(|p| {
                if p.contains('.') || pkg.is_empty() {
                    return p;
                }
                let qualified = format!("{}.{}", pkg, p);
                if self.classes.contains_key(&qualified) {
                    qualified
                } else {
                    p
                }
            });
        }
        Some(total)
    }

    /// (own field count, parent name, declaring package) for a class,
    /// following typedef aliases and the unambiguous bare-name form.
    fn resolve_class_entry(
        &mut self,
        class_name: &str,
        resolve_file: &dyn Fn(&str) -> Option<std::path::PathBuf>,
    ) -> Option<(usize, Option<String>, String)> {
        let mut name = class_name.to_string();
        for _ in 0..4 {
            self.ensure_indexed(&name, resolve_file);
            if let Some(class) = self.classes.get(&name) {
                return Some((
                    class.instance_field_count,
                    class.extends.clone(),
                    class.package.clone(),
                ));
            }
            match self.aliases.get(&name) {
                Some(target) => name = target.clone(),
                None => break,
            }
        }
        if !class_name.contains('.') {
            let candidates: Vec<&String> = self
                .bare_to_qualified
                .get(class_name)
                .map(|set| set.iter().collect())
                .unwrap_or_default();
            if let [only] = candidates.as_slice() {
                return self
                    .classes
                    .get(*only)
                    .map(|c| (c.instance_field_count, c.extends.clone(), c.package.clone()));
            }
        }
        None
    }

    /// Parse and index the file declaring `name` if it hasn't been seen.
    /// Failed resolutions are cached and never retried.
    fn ensure_indexed(
        &mut self,
        name: &str,
        resolve_file: &dyn Fn(&str) -> Option<std::path::PathBuf>,
    ) {
        if self.classes.contains_key(name)
            || self.aliases.contains_key(name)
            || self.parse_misses.contains(name)
        {
            return;
        }
        let parsed = resolve_file(name)
            .and_then(|path| std::fs::read_to_string(&path).ok().map(|src| (path, src)))
            .and_then(|(path, src)| {
                parser::haxe_parser::parse_haxe_file_with_config(
                    &path.to_string_lossy(),
                    &src,
                    true,
                    false,
                    &self.preprocessor,
                )
                .ok()
            });
        match parsed {
            Some(file) => self.index_file(&file),
            None => {
                self.parse_misses.insert(name.to_string());
            }
        }
        // The parse may still not declare `name` itself (e.g. the file only
        // holds a typedef for it) — record a miss so we don't re-parse.
        if !self.classes.contains_key(name) && !self.aliases.contains_key(name) {
            self.parse_misses.insert(name.to_string());
        }
    }
}
