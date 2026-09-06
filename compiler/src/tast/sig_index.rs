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
    /// Parameter count of a body-ful `function new` this class declares itself,
    /// recorded only for a non-`extern` `class`. Every other shape (abstract,
    /// extern, bodyless `new`, inherited ctor) is absent, because a `<C>.new`
    /// stub for those would dangle and SIGILL at the call.
    ctor_params: Option<usize>,
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
    /// Type name → declaring file, recorded by the import loader from paths it
    /// already resolved, so MIR lowering (which has no resolver) can read the
    /// declaration of a class whose file lowers later. Free to record; nothing
    /// is parsed until a `new` needs it. Membership also scopes the constructor
    /// query to this program's own import graph.
    known_files: BTreeMap<String, std::path::PathBuf>,
}

impl StaticSigIndex {
    /// Record where a type is declared, WITHOUT parsing it: one key and one
    /// path per import the loader has already resolved. First writer wins, so
    /// a bare name two packages both spell keeps the resolution the loader
    /// made.
    pub fn record_file(&mut self, name: &str, path: std::path::PathBuf) {
        if name.is_empty() {
            return;
        }
        self.known_files.entry(name.to_string()).or_insert(path);
    }

    /// Index the file declaring `class_name` via the loader's name→path table.
    /// Separate from `ensure_indexed`, which ast_lowering reaches with a real
    /// resolver — widening it there would retype method signatures frontend-wide.
    /// No-op once the name is known, known-missing, or absent from the table.
    pub fn ensure_indexed_from_known_files(&mut self, class_name: &str) {
        if self.classes.contains_key(class_name)
            || self.aliases.contains_key(class_name)
            || self.parse_misses.contains(class_name)
        {
            return;
        }
        // Cloned out first: nothing may borrow `self` across `ensure_indexed`,
        // which takes `&mut self`. The explicit `&dyn Fn` annotation forces the
        // higher-ranked closure signature at the coercion site.
        let Some(path) = self.known_file(class_name) else {
            return;
        };
        let resolve_file: &dyn Fn(&str) -> Option<std::path::PathBuf> =
            &move |_: &str| Some(path.clone());
        self.ensure_indexed(class_name, resolve_file);
    }

    /// The file the loader resolved for `class_name`, under the name it used:
    /// the qualified name, or the bare name when the path itself spells the
    /// package (`haxe/iterators/StringIterator.hx` for
    /// `haxe.iterators.StringIterator`). Never a bare name alone.
    fn known_file(&self, class_name: &str) -> Option<std::path::PathBuf> {
        if let Some(p) = self.known_files.get(class_name) {
            return Some(p.clone());
        }
        let (_, bare) = class_name.rsplit_once('.')?;
        let p = self.known_files.get(bare)?;
        let suffix = format!("{}.hx", class_name.replace('.', "/"));
        p.ends_with(std::path::Path::new(&suffix)).then(|| p.clone())
    }

    /// Declared constructor arity for a class indexed under EXACTLY this name.
    /// Strict on two axes, because the caller turns a `Some` into an allocation
    /// plus a named call that must resolve: exact name (no alias or bare-name
    /// matching), and the loader must have resolved it to a file of its own.
    /// `extern class String` records no arity, so it keeps the value wrap.
    pub fn declared_constructor_arity(&mut self, class_name: &str) -> Option<usize> {
        self.known_file(class_name)?;
        self.ensure_indexed_from_known_files(class_name);
        self.classes.get(class_name)?.ctor_params
    }

    /// The ancestor whose constructor `class_name` runs, with its arity.
    ///
    /// `class Sub extends Base {}` is constructible in Haxe and runs Base's
    /// constructor. The MIR walk for that reads the current module's HIR, so a
    /// subclass whose parent was declared elsewhere finds nothing and the call
    /// goes to a stub named after a constructor that does not exist. Names are
    /// what cross a module boundary, so walk the recorded `extends` here.
    pub fn constructor_owner(&mut self, class_name: &str) -> Option<(String, usize)> {
        let mut next = Some(class_name.to_string());
        let mut seen: BTreeSet<String> = BTreeSet::new();
        while let Some(name) = next {
            if !seen.insert(name.clone()) {
                break; // cyclic `extends` -- stop rather than spin
            }
            // The entry file is parsed but not registered as a known file, so
            // an already-indexed class counts even without one.
            if self.known_file(&name).is_some() {
                self.ensure_indexed_from_known_files(&name);
            }
            let class = self.classes.get(&name)?;
            if let Some(arity) = class.ctor_params {
                return Some((name, arity));
            }
            let (parent, pkg) = (class.extends.clone(), class.package.clone());
            // A bare `extends Base` names a type in the subclass's own package
            // first; fall back to the name as written so an imported parent
            // still resolves through the unambiguous bare form.
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
        None
    }

    /// Whether the class indexed under EXACTLY this name declares instance
    /// method `name` itself. Same strictness as `declared_constructor_arity`.
    pub fn declares_instance_method(&mut self, class_name: &str, name: &str) -> bool {
        if self.known_file(class_name).is_none() {
            return false;
        }
        self.ensure_indexed_from_known_files(class_name);
        self.classes
            .get(class_name)
            .is_some_and(|c| c.instances.contains_key(name))
    }

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
                // Only a class that will own a MIR constructor may record one.
                // `extern` is a modifier, not a variant, so `extern class Host`
                // arrives here yet has no MIR body; it keeps the value wrap,
                // which is what makes `new String(s)` work today.
                let record_ctor = !c
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, parser::Modifier::Extern));
                self.index_fields(package, &c.name, &c.fields, parent, record_ctor);
            }
            TypeDeclaration::Abstract(a) => {
                // `false`: an abstract must never record a constructor. Its
                // `new` is a value wrap over the underlying value, and a
                // cross-module `new SomeAbstract(x)` that reaches the fallback
                // in `lower_new` is lowered correctly by that wrap today.
                self.index_fields(package, &a.name, &a.fields, None, false);
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
        record_ctor: bool,
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
                // Recorded separately so the method tables stay keyed by
                // callable name; first declaration wins, like the tables below.
                // `body.is_some()` is load-bearing: a bodyless `new` is a
                // declaration with no definition, so its stub would never bind.
                let ctor_is_extern = field
                    .modifiers
                    .iter()
                    .any(|m| matches!(m, parser::Modifier::Extern));
                if record_ctor
                    && func.body.is_some()
                    && !ctor_is_extern
                    && entry.ctor_params.is_none()
                {
                    entry.ctor_params = Some(func.params.len());
                }
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

    /// Total instance slots for `class_name`: own declared fields plus every
    /// ancestor's. Used to size an allocation when a `new` lowers before its
    /// class does. An unresolvable ancestor yields None rather than a partial
    /// count, which would under-allocate and corrupt the heap.
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
