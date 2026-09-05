//! Finding what a file depends on and bringing those modules in.

use super::*;

impl CompilationUnit {

    /// Extract all class references from a Haxe AST file.
    /// This includes explicit imports, using statements, new expressions, and type annotations.
    pub(crate) fn extract_all_dependencies(ast: &parser::HaxeFile) -> Vec<String> {
        use parser::{BlockElement, ClassFieldKind, ExprKind, Type, TypeDeclaration};

        let mut deps = std::collections::BTreeSet::new();

        // 1. Explicit imports
        for import in &ast.imports {
            if !import.path.is_empty() {
                deps.insert(import.path.join("."));
            }
        }

        // 2. Using statements
        for using in &ast.using {
            if !using.path.is_empty() {
                deps.insert(using.path.join("."));
            }
        }

        // Helper to extract type references from a Type
        fn extract_type_deps(ty: &Type, deps: &mut std::collections::BTreeSet<String>) {
            match ty {
                Type::Path { path, params, .. } => {
                    // Only add if it looks like a class name (starts with uppercase)
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&path.name)
                        {
                            deps.insert(path.name.clone());
                        }
                    } else if !path.package.is_empty() {
                        // Qualified type path like sys.io.File
                        let mut full_path = path.package.clone();
                        full_path.push(path.name.clone());
                        deps.insert(full_path.join("."));
                    }
                    // Recurse into type parameters
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                }
                Type::Function { params, ret, .. } => {
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                    extract_type_deps(ret, deps);
                }
                Type::Anonymous { fields, .. } => {
                    for field in fields {
                        extract_type_deps(&field.type_hint, deps);
                    }
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from a block element
        fn extract_block_elem_deps(
            elem: &BlockElement,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match elem {
                BlockElement::Expr(e) => extract_expr_deps(e, deps),
                BlockElement::Import(imp) => {
                    if !imp.path.is_empty() {
                        deps.insert(imp.path.join("."));
                    }
                }
                BlockElement::Using(u) => {
                    if !u.path.is_empty() {
                        deps.insert(u.path.join("."));
                    }
                }
                BlockElement::Conditional(cond) => {
                    // Handle #if branch
                    for elem in &cond.if_branch.content {
                        extract_block_elem_deps(elem, deps);
                    }
                    // Handle #elseif branches
                    for branch in &cond.elseif_branches {
                        for elem in &branch.content {
                            extract_block_elem_deps(elem, deps);
                        }
                    }
                    // Handle #else branch
                    if let Some(else_body) = &cond.else_branch {
                        for elem in else_body {
                            extract_block_elem_deps(elem, deps);
                        }
                    }
                }
            }
        }

        // Helper to extract dependencies from an expression
        fn extract_expr_deps(expr: &parser::Expr, deps: &mut std::collections::BTreeSet<String>) {
            match &expr.kind {
                ExprKind::New {
                    type_path,
                    params,
                    args,
                } => {
                    // Extract class name from new expression
                    if type_path.package.is_empty() && !type_path.name.is_empty() {
                        let first_char = type_path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&type_path.name)
                        {
                            deps.insert(type_path.name.clone());
                            deps.insert(format!("new:{}", type_path.name));
                        }
                    } else if !type_path.package.is_empty() {
                        let mut full_path = type_path.package.clone();
                        full_path.push(type_path.name.clone());
                        deps.insert(full_path.join("."));
                        deps.insert(format!("new:{}", full_path.join(".")));
                    }
                    // Recurse into type params and args
                    for param in params {
                        extract_type_deps(param, deps);
                    }
                    for arg in args {
                        extract_expr_deps(arg, deps);
                    }
                }
                ExprKind::Call { expr, args } => {
                    extract_expr_deps(expr, deps);
                    for arg in args {
                        extract_expr_deps(arg, deps);
                    }
                }
                ExprKind::Field { expr, field, .. } => {
                    // If the object is a capitalized identifier, it may be a class/module
                    // reference (e.g. NativeStackTrace.exceptionStack()). Add it as a
                    // potential unqualified dep so load_imports_efficiently can resolve it
                    // via package-prefix fallback (tries haxe.X, haxe.ds.X, etc.).
                    if let ExprKind::Ident(name) = &expr.kind {
                        if name
                            .chars()
                            .next()
                            .map(|c| c.is_uppercase())
                            .unwrap_or(false)
                            && !is_stdtypes_ambient_name(name)
                        {
                            deps.insert(name.clone());
                        }
                    }
                    // Also try to extract qualified paths from nested field chains
                    // e.g. haxe.io.Bytes.ofString(...) → "haxe.io.Bytes"
                    // We check the full chain including current field — if the last
                    // component is uppercase, it's a package.Class pattern
                    if field
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false)
                    {
                        // Current field is uppercase (e.g. "Bytes" in haxe.io.Bytes)
                        // Try to build full qualified path from the expression chain + this field
                        let mut parts = Vec::new();
                        fn collect_parts_for_dep(e: &parser::Expr, parts: &mut Vec<String>) {
                            match &e.kind {
                                ExprKind::Ident(name) => parts.push(name.clone()),
                                ExprKind::Field { expr, field, .. } => {
                                    collect_parts_for_dep(expr, parts);
                                    parts.push(field.clone());
                                }
                                _ => {}
                            }
                        }
                        collect_parts_for_dep(expr, &mut parts);
                        parts.push(field.clone());
                        if parts.len() >= 2 {
                            deps.insert(parts.join("."));
                        }
                    }
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Index { expr, index } => {
                    extract_expr_deps(expr, deps);
                    extract_expr_deps(index, deps);
                }
                ExprKind::Unary { expr, .. } => {
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Binary { left, right, .. } => {
                    extract_expr_deps(left, deps);
                    extract_expr_deps(right, deps);
                }
                ExprKind::Assign { left, right, .. } => {
                    extract_expr_deps(left, deps);
                    extract_expr_deps(right, deps);
                }
                ExprKind::Ternary {
                    cond,
                    then_expr,
                    else_expr,
                } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(then_expr, deps);
                    extract_expr_deps(else_expr, deps);
                }
                ExprKind::Array(elems) => {
                    for elem in elems {
                        extract_expr_deps(elem, deps);
                    }
                }
                ExprKind::Block(elems) => {
                    for elem in elems {
                        extract_block_elem_deps(elem, deps);
                    }
                }
                ExprKind::Var {
                    type_hint, expr, ..
                }
                | ExprKind::Final {
                    type_hint, expr, ..
                } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                    if let Some(e) = expr {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::Return(Some(e)) | ExprKind::Throw(e) => {
                    extract_expr_deps(e, deps);
                }
                ExprKind::If {
                    cond,
                    then_branch,
                    else_branch,
                } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(then_branch, deps);
                    if let Some(e) = else_branch {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::While { cond, body } | ExprKind::DoWhile { body, cond } => {
                    extract_expr_deps(cond, deps);
                    extract_expr_deps(body, deps);
                }
                ExprKind::For { iter, body, .. } => {
                    extract_expr_deps(iter, deps);
                    extract_expr_deps(body, deps);
                }
                ExprKind::Try {
                    expr,
                    catches,
                    finally_block,
                } => {
                    extract_expr_deps(expr, deps);
                    for catch in catches {
                        if let Some(ty) = &catch.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        extract_expr_deps(&catch.body, deps);
                    }
                    if let Some(finally) = finally_block {
                        extract_expr_deps(finally, deps);
                    }
                }
                ExprKind::Cast { expr, type_hint } => {
                    extract_expr_deps(expr, deps);
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                }
                ExprKind::TypeCheck { expr, type_hint } => {
                    extract_expr_deps(expr, deps);
                    extract_type_deps(type_hint, deps);
                }
                ExprKind::Switch {
                    expr,
                    cases,
                    default,
                } => {
                    extract_expr_deps(expr, deps);
                    for case in cases {
                        // Extract from patterns (they may contain constructor references)
                        for pattern in &case.patterns {
                            extract_pattern_deps(pattern, deps);
                        }
                        if let Some(guard) = &case.guard {
                            extract_expr_deps(guard, deps);
                        }
                        extract_expr_deps(&case.body, deps);
                    }
                    if let Some(d) = default {
                        extract_expr_deps(d, deps);
                    }
                }
                ExprKind::Arrow { expr, .. } => {
                    extract_expr_deps(expr, deps);
                }
                ExprKind::Map(pairs) => {
                    for (k, v) in pairs {
                        extract_expr_deps(k, deps);
                        extract_expr_deps(v, deps);
                    }
                }
                ExprKind::Object(fields) => {
                    for field in fields {
                        extract_expr_deps(&field.expr, deps);
                    }
                }
                ExprKind::Function(func) => {
                    // Extract from function parameters and return type
                    for param in &func.params {
                        if let Some(ty) = &param.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        if let Some(default) = &param.default_value {
                            extract_expr_deps(default, deps);
                        }
                    }
                    if let Some(ret) = &func.return_type {
                        extract_type_deps(ret, deps);
                    }
                    if let Some(body) = &func.body {
                        extract_expr_deps(body, deps);
                    }
                }
                ExprKind::Paren(e)
                | ExprKind::Untyped(e)
                | ExprKind::Meta { expr: e, .. }
                | ExprKind::Macro(e)
                | ExprKind::Inline(e)
                | ExprKind::Reify(e) => {
                    extract_expr_deps(e, deps);
                }
                ExprKind::Tuple(elements) => {
                    for e in elements {
                        extract_expr_deps(e, deps);
                    }
                }
                ExprKind::ArrayComprehension { for_parts, expr } => {
                    for part in for_parts {
                        extract_expr_deps(&part.iter, deps);
                    }
                    extract_expr_deps(expr, deps);
                }
                ExprKind::MapComprehension {
                    for_parts,
                    key,
                    value,
                } => {
                    for part in for_parts {
                        extract_expr_deps(&part.iter, deps);
                    }
                    extract_expr_deps(key, deps);
                    extract_expr_deps(value, deps);
                }
                ExprKind::StringInterpolation(parts) => {
                    for part in parts {
                        if let parser::StringPart::Interpolation(e) = part {
                            extract_expr_deps(e, deps);
                        }
                    }
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from patterns (in switch cases)
        fn extract_pattern_deps(
            pattern: &parser::Pattern,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match pattern {
                parser::Pattern::Const(e) => extract_expr_deps(e, deps),
                parser::Pattern::Constructor { path, params } => {
                    // Constructor patterns reference enum/class types
                    if path.package.is_empty() && !path.name.is_empty() {
                        let first_char = path.name.chars().next();
                        if first_char.map(|c| c.is_uppercase()).unwrap_or(false)
                            && !is_stdtypes_ambient_name(&path.name)
                        {
                            deps.insert(path.name.clone());
                        }
                    } else if !path.package.is_empty() {
                        let mut full_path = path.package.clone();
                        full_path.push(path.name.clone());
                        deps.insert(full_path.join("."));
                    }
                    for param in params {
                        extract_pattern_deps(param, deps);
                    }
                }
                parser::Pattern::Array(patterns) | parser::Pattern::Or(patterns) => {
                    for p in patterns {
                        extract_pattern_deps(p, deps);
                    }
                }
                parser::Pattern::ArrayRest { elements, .. } => {
                    for p in elements {
                        extract_pattern_deps(p, deps);
                    }
                }
                parser::Pattern::Object { fields } => {
                    for (_, pattern) in fields {
                        extract_pattern_deps(pattern, deps);
                    }
                }
                parser::Pattern::Type { type_hint, .. } => {
                    extract_type_deps(type_hint, deps);
                }
                parser::Pattern::Extractor { expr, value } => {
                    extract_expr_deps(expr, deps);
                    extract_expr_deps(value, deps);
                }
                _ => {}
            }
        }

        // Helper to extract dependencies from class fields
        fn extract_field_deps(
            field: &parser::ClassField,
            deps: &mut std::collections::BTreeSet<String>,
        ) {
            match &field.kind {
                ClassFieldKind::Var {
                    type_hint, expr, ..
                }
                | ClassFieldKind::Final {
                    type_hint, expr, ..
                } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                    if let Some(e) = expr {
                        extract_expr_deps(e, deps);
                    }
                }
                ClassFieldKind::Property { type_hint, .. } => {
                    if let Some(ty) = type_hint {
                        extract_type_deps(ty, deps);
                    }
                }
                ClassFieldKind::Function(func) => {
                    for param in &func.params {
                        if let Some(ty) = &param.type_hint {
                            extract_type_deps(ty, deps);
                        }
                        if let Some(default) = &param.default_value {
                            extract_expr_deps(default, deps);
                        }
                    }
                    if let Some(ret) = &func.return_type {
                        extract_type_deps(ret, deps);
                    }
                    if let Some(body) = &func.body {
                        extract_expr_deps(body, deps);
                    }
                }
            }
        }

        // 3. Extract from type declarations (classes, interfaces, etc.)
        for decl in &ast.declarations {
            match decl {
                TypeDeclaration::Class(class_decl) => {
                    // Extract from extends clause
                    if let Some(extends) = &class_decl.extends {
                        extract_type_deps(extends, &mut deps);
                    }
                    // Extract from implements clause
                    for impl_type in &class_decl.implements {
                        extract_type_deps(impl_type, &mut deps);
                    }
                    // Extract from fields
                    for field in &class_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Interface(iface_decl) => {
                    // Extract from extends clause
                    for extends in &iface_decl.extends {
                        extract_type_deps(extends, &mut deps);
                    }
                    // Extract from fields
                    for field in &iface_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Typedef(typedef_decl) => {
                    extract_type_deps(&typedef_decl.type_def, &mut deps);
                }
                TypeDeclaration::Enum(enum_decl) => {
                    for ctor in &enum_decl.constructors {
                        for param in &ctor.params {
                            if let Some(ty) = &param.type_hint {
                                extract_type_deps(ty, &mut deps);
                            }
                        }
                    }
                }
                TypeDeclaration::Abstract(abstract_decl) => {
                    if let Some(ty) = &abstract_decl.underlying {
                        extract_type_deps(ty, &mut deps);
                    }
                    for ty in &abstract_decl.from {
                        extract_type_deps(ty, &mut deps);
                    }
                    for ty in &abstract_decl.to {
                        extract_type_deps(ty, &mut deps);
                    }
                    for field in &abstract_decl.fields {
                        extract_field_deps(field, &mut deps);
                    }
                }
                TypeDeclaration::Conditional(cond) => {
                    // Handle conditional compilation blocks
                    // Handle #if branch
                    for inner_decl in &cond.if_branch.content {
                        if let TypeDeclaration::Class(c) = inner_decl {
                            for field in &c.fields {
                                extract_field_deps(field, &mut deps);
                            }
                        }
                    }
                    // Handle #elseif branches
                    for branch in &cond.elseif_branches {
                        for inner_decl in &branch.content {
                            if let TypeDeclaration::Class(c) = inner_decl {
                                for field in &c.fields {
                                    extract_field_deps(field, &mut deps);
                                }
                            }
                        }
                    }
                    // Handle #else branch
                    if let Some(else_body) = &cond.else_branch {
                        for inner_decl in else_body {
                            if let TypeDeclaration::Class(c) = inner_decl {
                                for field in &c.fields {
                                    extract_field_deps(field, &mut deps);
                                }
                            }
                        }
                    }
                }
            }
        }

        // StdTypes.hx contributes top-level prelude types. They are not files
        // the user imports, so do not feed bare `Iterator`, `Null`, `Bool`, etc.
        // into load_imports_efficiently where they become failed path guesses.
        deps.retain(|d| !Self::is_bare_stdtypes_prelude_dependency(d));

        // Qualify bare type names with the file's package.
        // e.g., if File.hx has `package sys.io;` and references `FileInput`,
        // also add `sys.io.FileInput` so the import loader can find it.
        if let Some(package) = &ast.package {
            if !package.path.is_empty() {
                let package_prefix = package.path.join(".");
                let qualified: Vec<String> = deps
                    .iter()
                    .filter(|d| {
                        !d.contains('.')
                            && d.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    })
                    .map(|d| format!("{}.{}", package_prefix, d))
                    .collect();
                for q in qualified {
                    deps.insert(q);
                }
            }
        }

        // A type this file declares itself is not a dependency. Leaving it in
        // sends an undotted name to the import loader, which guesses a package
        // for it — so a file with its own `Resource` pulls in `haxe.Resource`
        // and reports that an unrelated module failed to compile.
        let own_types: std::collections::BTreeSet<String> = ast
            .declarations
            .iter()
            .filter_map(|decl| match decl {
                TypeDeclaration::Class(c) => Some(c.name.clone()),
                TypeDeclaration::Interface(i) => Some(i.name.clone()),
                TypeDeclaration::Enum(e) => Some(e.name.clone()),
                TypeDeclaration::Abstract(a) => Some(a.name.clone()),
                TypeDeclaration::Typedef(t) => Some(t.name.clone()),
                _ => None,
            })
            .collect();
        deps.retain(|d| !own_types.contains(d));
        deps.retain(|d| !Self::is_bare_stdtypes_prelude_dependency(d));

        let mut result: Vec<String> = deps.into_iter().collect();
        result.sort();
        result
    }


    /// Load imports efficiently by pre-collecting all dependencies and compiling in topological order.
    /// This avoids the fail-retry pattern that causes exponential recompilation.
    pub fn load_imports_efficiently(&mut self, imports: &[String]) -> Result<(), String> {
        use std::collections::{BTreeMap, BTreeSet, VecDeque};

        // Step 1: Collect all files and their dependencies by parsing (not compiling)
        // Use BTreeMap for deterministic iteration order
        // and causes non-deterministic import base offsets, leading to different function
        // IDs, different inlining decisions, and ultimately wrong optimized MIR.
        let mut all_files: BTreeMap<String, (PathBuf, String, Vec<String>)> = BTreeMap::new();
        // Which files each file CONSTRUCTS, split off from its dependencies:
        // a cycle is broken by emitting the constructed class first.
        let mut constructs: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut to_process: VecDeque<String> = VecDeque::new();
        for name in imports {
            if is_stdtypes_ambient_import(name) {
                continue;
            }
            to_process.push_back(name.clone());
        }
        let mut visited: BTreeSet<String> = BTreeSet::new();

        let t_discover = profile_timer(self.config.profile_typecheck);
        while let Some(qualified_path) = to_process.pop_front() {
            if is_stdtypes_ambient_import(&qualified_path) || visited.contains(&qualified_path) {
                continue;
            }
            visited.insert(qualified_path.clone());

            // Resolve to file path (use _force variant to bypass BLADE cache's
            // is_file_loaded check — BLADE pre-registers symbols but doesn't preserve
            // full TAST state needed for generic instantiation and method resolution)
            let resolved = self
                .namespace_resolver
                .resolve_qualified_path_to_file_force(&qualified_path);
            let file_path = if let Some(path) = resolved {
                path
            } else if !qualified_path.contains('.') {
                // Try common prefixes for unqualified names
                let prefixes = [
                    "haxe.iterators",
                    "haxe.ds",
                    "haxe",
                    "sys.thread",
                    "sys",
                    "haxe.exceptions",
                    "haxe.io",
                ];
                let mut found = None;
                for prefix in &prefixes {
                    let full = format!("{}.{}", prefix, qualified_path);
                    if let Some(path) = self
                        .namespace_resolver
                        .resolve_qualified_path_to_file_force(&full)
                    {
                        found = Some(path);
                        break;
                    }
                }
                match found {
                    Some(p) => p,
                    None => continue, // Skip unresolvable imports
                }
            } else {
                continue; // Skip unresolvable
            };

            // Deduplicate by file path — the same file can appear under different
            // qualified names (e.g., "BalancedTree" and "haxe.ds.BalancedTree")
            let file_path_str = file_path.to_string_lossy().to_string();
            if all_files
                .values()
                .any(|(p, _, _)| p.to_string_lossy() == file_path_str)
            {
                continue;
            }

            // Read and parse to extract imports
            let source = match std::fs::read_to_string(&file_path) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let filename = file_path_str;
            // Parsed with the compile's own defines. The parser's defaults lack
            // `eval`, so a reference inside `#if eval` — StringMap's
            // `new MapKeyValueIterator(this)` — never reached the dependency
            // graph, and the iterator was lowered after its constructor site.
            let deps = match parser::haxe_parser::parse_haxe_file_with_config(
                &filename,
                &source,
                false,
                false,
                &self.preprocessor_config(),
            ) {
                Ok(ast) => Self::extract_all_dependencies(&ast),
                Err(_) => Vec::new(),
            };
            let (ctor_marks, deps): (Vec<String>, Vec<String>) =
                deps.into_iter().partition(|d| d.starts_with("new:"));
            constructs.insert(
                qualified_path.clone(),
                ctor_marks.iter().map(|d| d["new:".len()..].to_string()).collect(),
            );
            // Queue dependencies for processing
            for dep in &deps {
                if is_stdtypes_ambient_import(dep) {
                    continue;
                }
                if !visited.contains(dep) {
                    to_process.push_back(dep.clone());
                }
            }

            // Where each type is declared, recorded before anything compiles:
            // MIR lowering has no namespace resolver, so this is how
            // `StaticSigIndex` reads the declaration of a class whose own file
            // lowers later. Path only — no parse, no I/O, order untouched.
            self.static_sig_index
                .borrow_mut()
                .record_file(&qualified_path, file_path.clone());

            all_files.insert(qualified_path.clone(), (file_path, source, deps));
        }

        // Debug: log collected files
        if !all_files.is_empty() {
            debug!(
                "[IMPORT_LOAD] Collected {} files for import",
                all_files.len()
            );
        }
        if self.config.profile_typecheck {
            self.typecheck_timings.imports_collected += all_files.len();
        }

        add_profile_ms(&mut self.typecheck_timings.import_discover_ms, t_discover);

        // Step 2: Topological sort using Kahn's algorithm
        let t_toposort = profile_timer(self.config.profile_typecheck);
        // Use BTreeMap for deterministic iteration order
        let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
        let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();

        // Build reverse map: bare class name → qualified name for same-package deps
        let bare_to_qualified: BTreeMap<String, String> = all_files
            .keys()
            .filter_map(|qn| {
                let short = qn.rsplit('.').next()?;
                Some((short.to_string(), qn.clone()))
            })
            .collect();

        for (name, (_, _, deps)) in &all_files {
            in_degree.entry(name.clone()).or_insert(0);
            for dep in deps {
                // Skip self-dependencies (class referencing itself in its own file)
                if dep == name {
                    continue;
                }
                // Check both qualified name ("sim.Point2D") and bare name ("Point2D")
                let resolved_dep = if all_files.contains_key(dep) {
                    Some(dep.clone())
                } else {
                    bare_to_qualified.get(dep).cloned()
                };
                if let Some(resolved) = resolved_dep {
                    if resolved != *name {
                        graph
                            .entry(resolved.clone())
                            .or_default()
                            .push(name.clone());
                        *in_degree.entry(name.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        if std::env::var_os("RAYZOR_IMPORT_GRAPH").is_some() {
            for (dep, dependents) in &graph {
                eprintln!("[import-graph] {} -> {}", dep, dependents.join(", "));
            }
        }
        let mut queue: VecDeque<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();

        let mut compile_order: Vec<String> = Vec::new();

        while let Some(name) = queue.pop_front() {
            compile_order.push(name.clone());
            if let Some(dependents) = graph.get(&name) {
                for dep in dependents {
                    if let Some(deg) = in_degree.get_mut(dep) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(dep.clone());
                        }
                    }
                }
            }
        }

        // Handle cycle: if compile_order doesn't include all files, some are stuck in a cycle.
        // Append remaining files in any order (they'll still compile, just without guaranteed dep order).
        if compile_order.len() < all_files.len() {
            let stuck_count = all_files.len() - compile_order.len();
            if stuck_count > 0 {
                debug!(
                    "Cycle detected, {} files stuck in dependency cycle. Forcing compilation.",
                    stuck_count
                );
                let in_order: std::collections::BTreeSet<_> =
                    compile_order.iter().cloned().collect();
                // Mini topological sort for stuck files: repeatedly emit files
                // whose deps (among remaining stuck files) are all already emitted.
                let mut stuck: BTreeMap<String, Vec<String>> = BTreeMap::new();
                for name in all_files.keys() {
                    if in_order.contains(name) {
                        continue;
                    }
                    let deps_in_stuck: Vec<String> = all_files
                        .get(name)
                        .map(|(_, _, deps)| {
                            deps.iter()
                                .filter_map(|d| {
                                    let resolved = if all_files.contains_key(d) {
                                        Some(d.clone())
                                    } else {
                                        bare_to_qualified.get(d).cloned()
                                    };
                                    resolved.filter(|r| {
                                        r != name
                                            && !in_order.contains(r)
                                            && all_files.contains_key(r)
                                    })
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    stuck.insert(name.clone(), deps_in_stuck);
                }
                let mut emitted: BTreeSet<String> = BTreeSet::new();
                loop {
                    let ready: Vec<String> = stuck
                        .iter()
                        .filter(|(_, deps)| deps.iter().all(|d| emitted.contains(d)))
                        .map(|(name, _)| name.clone())
                        .collect();
                    if !ready.is_empty() {
                        for name in &ready {
                            compile_order.push(name.clone());
                            emitted.insert(name.clone());
                            stuck.remove(name);
                        }
                        continue; // Check for more ready files
                    }
                    // No ready file — break a cycle by emitting the one with the
                    // FEWEST still-unemitted stuck-deps (ties: name order). This
                    // emits leaf-like dependencies first so a constructor's
                    // callee (e.g. `LlamaModel`, built by `LlamaArch.build` via
                    // `new LlamaModel()`) lands before its caller — otherwise the
                    // caller can't resolve the callee's constructor and leaves the
                    // object unconstructed.
                    let victim = stuck
                        .iter()
                        .map(|(name, deps)| {
                            let outstanding = deps.iter().filter(|d| !emitted.contains(*d)).count();
                            // Among equals, the class that is constructed goes
                            // before the class that constructs it, so its
                            // constructor exists when the `new` lowers.
                            let builds_a_stuck_file = constructs
                                .get(name)
                                .map(|cs| {
                                    cs.iter().any(|c| {
                                        let r = if all_files.contains_key(c) {
                                            Some(c.clone())
                                        } else {
                                            bare_to_qualified.get(c).cloned()
                                        };
                                        r.is_some_and(|r| r != *name && stuck.contains_key(&r))
                                    })
                                })
                                .unwrap_or(false);
                            (outstanding, builds_a_stuck_file, name.clone())
                        })
                        .min()
                        .map(|(_, _, name)| name);
                    if let Some(name) = victim {
                        compile_order.push(name.clone());
                        emitted.insert(name.clone());
                        stuck.remove(&name);
                        continue; // Re-check after cycle break
                    }
                    break; // All emitted
                }
            }
        }

        add_profile_ms(&mut self.typecheck_timings.import_toposort_ms, t_toposort);

        let t_import_compile = profile_timer(self.config.profile_typecheck);
        if std::env::var_os("RAYZOR_IMPORT_GRAPH").is_some() {
            for (i, n) in compile_order.iter().enumerate() {
                eprintln!("[import-order] {:>3} {}", i, n);
            }
        }
        // Step 3: Compile in topological order with retry for files that fail
        // due to unresolved symbols (dependency ordering issues from cycles).
        debug!(
            "[IMPORT_LOAD] Compiling {} stdlib files: {:?}",
            compile_order.len(),
            compile_order
        );
        // Snapshot the diagnostics length before each first-pass attempt so
        // we can discard import errors that get resolved by the retry pass.
        // Without this, every transient dependency-ordering failure surfaces
        // (e.g. `Cannot find name 'FPHelper'` from haxe.io.Input compiled
        // before haxe.io.FPHelper) even though the retry succeeds.
        let first_pass_snapshot = self.collected_diagnostics.len();
        let mut retry_queue: Vec<(String, PathBuf, String, Vec<String>, usize)> = Vec::new();
        for name in compile_order {
            if let Some((file_path, source, deps)) = all_files.remove(&name) {
                let diag_snapshot = self.collected_diagnostics.len();
                if !self.try_compile_import(&name, &file_path, &source, deps.clone()) {
                    retry_queue.push((name, file_path, source, deps, diag_snapshot));
                } else {
                    // Success: any diagnostics pushed during this attempt were
                    // recovered (e.g. from a partial parse) — keep them.
                }
            }
        }

        // Retry failed files until a pass resolves nothing new. Their
        // dependencies get registered as earlier files in the queue succeed —
        // but a DEEP import chain (A imports B imports C, all failing the first
        // pass on ordering) needs as many passes as its depth. The prior code
        // did a SINGLE retry, so it cleared only one level: a long chain like
        // `Main -> GGUFLoader -> GGUFReader.TensorInfo -> ...` left the tail
        // stranded as an empty MIR module whose methods then trap as forward-ref
        // stubs (udf #0xc11f / wasm unreachable) at unrelated call sites. Each
        // pass discards transient ordering errors (truncate to
        // first_pass_snapshot); only the survivors of a no-progress pass are
        // surfaced as genuine failures. A failed attempt pushes no MIR (the
        // pop+push is in try_compile_import's Ok branch), so re-attempting is
        // safe and never duplicates a module.
        let mut pending: Vec<(String, PathBuf, String, Vec<String>)> = retry_queue
            .into_iter()
            .map(|(n, p, s, d, _snap)| (n, p, s, d))
            .collect();
        let final_failures: Vec<String> = loop {
            self.collected_diagnostics.truncate(first_pass_snapshot);
            let before = pending.len();
            let mut next = Vec::new();
            for (name, file_path, source, deps) in std::mem::take(&mut pending) {
                if !self.try_compile_import(&name, &file_path, &source, deps.clone()) {
                    next.push((name, file_path, source, deps));
                }
            }
            // No progress: the survivors are genuine failures; their errors
            // (pushed after first_pass_snapshot during this pass) are kept.
            if next.len() == before {
                break next.into_iter().map(|(n, ..)| n).collect();
            }
            pending = next;
        };

        // LOUD-FAIL only the genuine FINAL failures — modules still unresolved
        // after the retry loop converged. Transient first-pass ordering failures
        // (a module compiled before its dependency) are NOT reported here, since
        // a later pass resolved them. Each call into a truly-failed module lowers
        // to a forward-ref trap stub (udf #0xc11f / wasm `unreachable`), so this
        // is the difference between a clean run and a silent SIGILL.
        add_profile_ms(
            &mut self.typecheck_timings.import_compile_ms,
            t_import_compile,
        );
        for name in &final_failures {
            if let Some(errs) = self.last_import_errors.get(name) {
                eprintln!(
                    "error[IMPORT]: imported module `{}` failed to compile ({} error(s)); \
                     calls into it will trap at runtime. First errors:",
                    name,
                    errs.len()
                );
                for line in errs.iter().take(8) {
                    eprintln!("    {}", line);
                }
            }
        }

        // Fixup pass: resolve stale cross-module refs that couldn't be resolved during
        // renumbering because the target module hadn't been loaded yet (ordering issue
        // with blade cache). Now that ALL modules are loaded, stdlib_function_name_map
        // is complete and we can resolve any remaining stale refs.
        self.fixup_stale_cross_module_refs();
        self.fixup_stale_constructor_ids();
        self.fixup_stale_method_ids();

        Ok(())
    }


    /// Try to compile a single import file. Returns true on success, false on failure.
    pub(crate) fn try_compile_import(
        &mut self,
        name: &str,
        file_path: &Path,
        source: &str,
        deps: Vec<String>,
    ) -> bool {
        let filename = file_path.to_string_lossy().to_string();

        // Skip if already compiled
        if self.compiled_files.contains_key(&filename) {
            if self.config.profile_typecheck {
                self.typecheck_timings.import_already_compiled += 1;
            }
            return true;
        }

        // Mark as loaded
        self.namespace_resolver
            .mark_file_loaded(file_path.to_path_buf());

        // Phase 2: capture raw AST of imports containing macros so that
        // cross-file macro discovery (e.g. `import tink.Json; tink.Json.parse(...)`)
        // works even when the import is served from BLADE cache. This is gated
        // by a cheap substring check to avoid parsing every stdlib file; only
        // files containing macro definitions need to be kept around for the
        // expander's dependency scan.
        if source.contains("macro ") || source.contains("@:build") {
            if let Ok(raw_ast) = self.parse_file(&filename, source) {
                self.loaded_import_haxe_files.push(raw_ast);
            }
        }

        // Try BLADE cache first. Typedef-only modules are cacheable now that
        // BLADE restores type aliases as first-class type-system symbols; their
        // dependencies are still walked by the import loader before consumers
        // resolve the alias target in the current compilation context.
        let source_has_typedef = source.contains("typedef ");
        let cache_hit = if self.config.enable_cache {
            let t_cache_load = profile_timer(self.config.profile_typecheck);
            let hit = self.try_load_import_from_cache(&filename, source);
            add_profile_ms(
                &mut self.typecheck_timings.import_cache_load_ms,
                t_cache_load,
            );
            if self.config.profile_typecheck {
                if hit {
                    self.typecheck_timings.import_cache_hits += 1;
                } else {
                    self.typecheck_timings.import_cache_misses += 1;
                }
            }
            hit
        } else {
            false
        };

        if cache_hit {
            // BLADE cache hit means compile_file_with_shared_state_ex →
            // compile_ast_with_shared_state was NOT called for this file,
            // so file_id_by_filename / file_source_by_filename never got
            // an entry. Allocate them now from the in-memory `source` so
            // the renderer can still locate this file by file_id and
            // resolve its byte_offsets against the same bytes the cached
            // MIR's spans were computed against.
            let file_id_u32 = *self
                .file_id_by_filename
                .entry(filename.to_string())
                .or_insert_with(|| {
                    let id = self.next_file_id;
                    self.next_file_id += 1;
                    id
                });
            let _ = file_id_u32;
            self.file_source_by_filename
                .entry(filename.to_string())
                .or_insert_with(|| source.to_string());
            // For the same reason, nothing fed this file's DECLARATIONS to the
            // static-signature index either, and that index is what types a
            // field access or a call into the module. Without it a cached
            // import contributes symbols and MIR but nothing that can answer
            // "does this class have a field `low`", so accesses into it fail
            // with E0100 and every call becomes a runtime trap -- while the
            // same program built with --no-cache compiles clean.
            //
            // The default stdlib imports are re-parsed for exactly this reason
            // where the manifest is loaded; a cached import needs it too, and
            // there the set cannot be known in advance. Parsing is a fraction
            // of the compile the cache just saved, and only files actually
            // imported pay it.
            let _ = self.parse_file(&filename, source);
            return true;
        }

        // Never skip pre-registration for import files — they always need
        // full type registration to make extern class fields visible.
        let is_stdlib = false;
        // User package imports skip the stdlib merge — it happens once in the main file's
        // compilation. This prevents stdlib functions from overwriting user functions by bare name.
        // Stdlib imports (EReg, StringTools, etc.) still need the merge because their Haxe
        // source has placeholder method bodies that must be replaced by MIR wrappers.
        let skip_stdlib_merge = !is_stdlib;
        let t_compile_call = profile_timer(self.config.profile_typecheck);
        let compile_outcome =
            self.compile_file_with_shared_state_ex(&filename, source, is_stdlib, skip_stdlib_merge);
        add_profile_ms(
            &mut self.typecheck_timings.import_compile_call_ms,
            t_compile_call,
        );
        match compile_outcome {
            Ok(typed_file) => {
                if self.config.profile_typecheck {
                    self.typecheck_timings.import_fresh_compiles += 1;
                    if source_has_typedef {
                        self.typecheck_timings.import_typedef_fresh += 1;
                    }
                }
                // Extract inline var constants before consuming the TypedFile.
                // These are stored in BLADE cache and in global_inline_vars for
                // cross-file static inline var resolution (e.g., Key.ESCAPE).
                let inline_vars = Self::extract_inline_vars_from_typed_file(
                    &typed_file,
                    &self.symbol_table,
                    &self.string_interner,
                );
                self.store_inline_vars(&inline_vars);

                // Register extern class methods as plugin mappings so MIR lowerer
                // can resolve them (same as rpkg NativePlugin does).
                self.register_extern_methods_from_typed_file(&typed_file);

                self.loaded_stdlib_typed_files.push(typed_file);

                // Move the MIR from mir_modules to import_mir_modules.
                if let Some(mir_arc) = self.mir_modules.pop() {
                    // Save to BLADE cache before renumbering
                    if self.config.enable_cache {
                        let type_info = self.last_compiled_type_info.take();
                        let mut cached_maps = self.last_compiled_cached_maps.take();
                        // Append inline vars to cached maps for cache persistence
                        if let Some(ref mut maps) = cached_maps {
                            maps.inline_vars = inline_vars;
                        }
                        let t_cache_save = profile_timer(self.config.profile_typecheck);
                        self.save_blade_cached(
                            &filename,
                            source,
                            &mir_arc,
                            deps,
                            type_info,
                            cached_maps,
                        );
                        add_profile_ms(
                            &mut self.typecheck_timings.import_cache_save_ms,
                            t_cache_save,
                        );
                    }

                    // Track which import functions are source-level declarations
                    // (methods, constructors) vs generated MIR wrappers.
                    // Only for USER PACKAGES — stdlib files (EReg, StringTools, etc.) have
                    // placeholder method bodies that must be replaced by stdlib MIR wrappers.
                    let own_ids = self.last_compiled_own_func_ids.take().unwrap_or_default();
                    let is_user_package = !filename.contains("haxe-std");
                    if is_user_package {
                        let import_base: u32 =
                            100_000 + (self.import_mir_modules.len() as u32 * 10_000);
                        for old_id in &own_ids {
                            let new_id = crate::ir::IrFunctionId(old_id.0 + import_base);
                            self.import_own_func_ids.insert(new_id);
                        }
                    }

                    self.renumber_and_push_import_mir((*mir_arc).clone());
                } else if self.config.enable_cache {
                    let type_info = self.last_compiled_type_info.take();
                    let _ = self.last_compiled_cached_maps.take();
                    if let Some(type_info) = type_info {
                        let module_name = std::path::Path::new(&filename)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .unwrap_or("stdlib")
                            .to_string();
                        let empty_mir = crate::ir::IrModule::new(module_name, filename.clone());
                        self.save_blade_cached(
                            &filename,
                            source,
                            &empty_mir,
                            deps,
                            Some(type_info),
                            None,
                        );
                    }
                }
                true
            }
            Err(errors) => {
                // LOUD-FAIL: an imported module that fails to compile yields an
                // empty MIR, so every call into it lowers to a forward-ref stub
                // that traps (udf #0xc11f / wasm `unreachable`) at call time with
                // NO diagnostic. Stash this attempt's errors keyed by module name
                // — try_compile_import runs once per retry pass, so printing here
                // would over-report transient ordering failures that a later pass
                // resolves. The retry loop surfaces only the genuine FINAL
                // survivors (see load_imports_efficiently) from this map.
                self.last_import_errors.insert(
                    name.to_string(),
                    errors
                        .iter()
                        .map(|e| {
                            format!("{} ({}:{})", e.message, e.location.line, e.location.column)
                        })
                        .collect(),
                );
                // Surface import failures via the regular diagnostic
                // pipeline. Previously these went only to debug-log and the
                // caller saw "false" — so a parse/type error in an imported
                // file (e.g. `var x = 1e-5` failing to lex as a Float)
                // silently produced an empty MIR module. Downstream lookups
                // for the imported class's methods then returned forward-ref
                // stubs that ended in `unreachable`, surfacing as a SIGILL
                // / silent exit at unrelated call sites in the importer.
                debug!(
                    "[IMPORT_LOAD] Failed to compile {}: {} error(s)",
                    name,
                    errors.len()
                );
                for e in &errors {
                    debug!("  - {}", e.message);
                    // Convert CompilationError → Diagnostic and add to the
                    // collected pool so the runner prints it.
                    let span = if e.location.is_valid() {
                        let pos = diagnostics::SourcePosition::new(
                            e.location.line as usize,
                            e.location.column as usize,
                            e.location.byte_offset as usize,
                        );
                        let end_pos = diagnostics::SourcePosition::new(
                            e.location.line as usize,
                            (e.location.column + 1) as usize,
                            (e.location.byte_offset + 1) as usize,
                        );
                        diagnostics::SourceSpan::new(pos, end_pos, diagnostics::FileId::new(0))
                    } else {
                        diagnostics::SourceSpan::new(
                            diagnostics::SourcePosition::new(0, 0, 0),
                            diagnostics::SourcePosition::new(0, 1, 1),
                            diagnostics::FileId::new(0),
                        )
                    };
                    self.collected_diagnostics.push(diagnostics::Diagnostic {
                        severity: diagnostics::DiagnosticSeverity::Error,
                        code: Some(format!("IMPORT[{}]", name)),
                        message: format!("imported file {}: {}", name, e.message),
                        span,
                        labels: Vec::new(),
                        suggestions: Vec::new(),
                        notes: Vec::new(),
                        help: Vec::new(),
                    });
                }
                false
            }
        }
    }


    /// Try to load an import file from BLADE cache.
    /// Returns true if cache hit (MIR loaded + symbols registered), false if miss.
    pub(crate) fn try_load_import_from_cache(&mut self, filename: &str, source: &str) -> bool {
        // Try to load from BLADE cache
        let (mir, _metadata, symbols, cached_maps) =
            match self.try_load_blade_cached_full(filename, source) {
                Some(data) => data,
                None => return false,
            };

        // We need both type info and cached maps for a full cache restore.
        // Typedef/extern-only imports can produce no MIR at all; for those,
        // symbols-only + empty MIR is enough to avoid redoing TAST/HIR work
        // when the entry module recompiles.
        let (symbols, cached_maps) = match (symbols, cached_maps) {
            (Some(s), Some(m)) => (s, Some(m)),
            (Some(s), None) if mir.functions.is_empty() => (s, None),
            _ => {
                debug!("[BLADE] Cache hit but missing type info/maps: {}", filename);
                return false;
            }
        };

        debug!(
            "[BLADE] Import cache hit: {} ({} functions, {} fields, {} class sizes)",
            filename,
            cached_maps.as_ref().map(|m| m.functions.len()).unwrap_or(0),
            cached_maps.as_ref().map(|m| m.fields.len()).unwrap_or(0),
            cached_maps
                .as_ref()
                .map(|m| m.class_sizes.len())
                .unwrap_or(0)
        );

        // Step 1: Register symbols from type info (restores type system state)
        let registered = self.register_symbols_from_type_info(&symbols);

        let Some(cached_maps) = cached_maps else {
            return true;
        };

        // Step 2: Rebuild MIR-level maps from cached maps using fresh IDs.
        // A reference this context cannot resolve makes the entry unusable:
        // loading it anyway is how a cached module comes back subtly different
        // from the one that was cached. Decline it and lower from source.
        let dropped = self.restore_cached_maps(&cached_maps, &registered);
        if dropped > 0 {
            // Measured across the standard library, this fires on most
            // restores: the cache routinely loads modules missing references
            // it could not resolve. Declining them is correct and costs a full
            // recompile of nearly everything — 17ms becomes 880ms — so until
            // the underlying resolution gaps are closed it stays opt-in, and
            // says so rather than passing silently.
            debug!(
                "[BLADE] {} unresolved cross-references restoring {}",
                dropped, filename
            );
            if std::env::var_os("RAYZOR_STRICT_BLADE").is_some() {
                return false;
            }
        }

        // Step 3: Build name-based function map from MIR
        // Use qualified names to avoid collisions (e.g., "current" matching
        // both ArrayIterator.current field and Thread.current method)
        for (func_id, func) in &mir.functions {
            if !func.cfg.blocks.is_empty() {
                // Prefer qualified_name (e.g., "ArrayIterator.hasNext") over bare name ("hasNext")
                let map_name = func.qualified_name.as_deref().unwrap_or(&func.name);
                self.stdlib_function_name_map
                    .insert(map_name.to_string(), *func_id);
            }
        }

        // Step 4: Restore inline var constants from cache
        if !cached_maps.inline_vars.is_empty() {
            self.store_inline_vars(&cached_maps.inline_vars);
        }

        // Step 4.5: Mark this import's SOURCE-DECLARED functions (methods +
        // constructors from cached_maps.functions) as OWNED, exactly like
        // the fresh-compile path does via last_compiled_own_func_ids.
        // Without this, a cache-loaded user module's methods (e.g.
        // KVCache.append) are unprotected at the stdlib merge: the merge
        // deletes/rewrites them, fresh modules lowering against them fall
        // through to a degenerate bare-name extern stub, and the JIT panics
        // at finalize with "can't resolve symbol append" (the AOT link
        // fails on the same bare symbol). This was the touch-entry-file /
        // edit-one-dep mixed-cache crash.
        //
        // Scope strictly to DECLARED functions: marking every non-empty-CFG
        // function also protects generated stdlib wrapper placeholders
        // inside the user module, which then shadow the real stdlib MIR
        // after the merge (first symptom: cached StringMap.get dispatching
        // into a stub — "missing meta key 'general.architecture'").
        // Stdlib files keep placeholder method bodies the stdlib wrappers
        // must replace, so the guard stays user-packages-only, mirroring
        // the fresh path.
        let is_user_package = !filename.contains("haxe-std");
        if is_user_package {
            let import_base: u32 = 100_000 + (self.import_mir_modules.len() as u32 * 10_000);
            for entry in &cached_maps.functions {
                let new_id = crate::ir::IrFunctionId(entry.func_id + import_base);
                self.import_own_func_ids.insert(new_id);
            }
        }

        // Step 5: Renumber and push to import_mir_modules
        self.renumber_and_push_import_mir(mir);

        true
    }


    /// Load a single file on-demand for import resolution (legacy - uses retry pattern)
    /// Prefer load_imports_efficiently for batch loading
    pub fn load_import_file(&mut self, qualified_path: &str) -> Result<(), String> {
        self.load_import_file_recursive(qualified_path, 0)
    }


    /// Internal recursive function for loading files with dependency resolution
    /// Max depth prevents infinite loops in circular dependencies
    pub(crate) fn load_import_file_recursive(
        &mut self,
        qualified_path: &str,
        depth: usize,
    ) -> Result<(), String> {
        const MAX_DEPTH: usize = 10;

        if depth > MAX_DEPTH {
            return Err(format!(
                "Maximum dependency depth ({}) exceeded for: {}",
                MAX_DEPTH, qualified_path
            ));
        }

        // Resolve the qualified path to a filesystem path
        // If not found directly, try common stdlib package prefixes for unqualified names
        let file_path = if let Some(path) = self
            .namespace_resolver
            .resolve_qualified_path_to_file(qualified_path)
        {
            path
        } else if !qualified_path.contains('.') {
            // Unqualified name - try common stdlib packages
            let prefixes = vec![
                "haxe.iterators",
                "haxe.ds",
                "haxe",
                "sys.thread",
                "sys",
                "haxe.exceptions",
                "haxe.io",
            ];
            let mut found_path = None;
            for prefix in &prefixes {
                let qualified = format!("{}.{}", prefix, qualified_path);
                if let Some(path) = self
                    .namespace_resolver
                    .resolve_qualified_path_to_file(&qualified)
                {
                    found_path = Some(path);
                    break;
                }
            }
            found_path.ok_or_else(|| format!("Could not resolve import: {}", qualified_path))?
        } else {
            // Check if the file is already loaded (resolve returns None for loaded files)
            if self
                .namespace_resolver
                .is_qualified_path_loaded(qualified_path)
            {
                return Ok(());
            }
            return Err(format!("Could not resolve import: {}", qualified_path));
        };

        // Skip if already loaded - this prevents redundant re-compilation
        if self.namespace_resolver.is_file_loaded(&file_path) {
            return Ok(());
        }

        // Mark as loaded BEFORE compiling to prevent recursive loading
        self.namespace_resolver.mark_file_loaded(file_path.clone());

        // Read the file
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("Failed to read {:?}: {}", file_path, e))?;

        let filename = file_path.to_string_lossy().to_string();

        // Try to compile - if it fails due to missing dependencies, extract and load them
        match self.compile_file_with_shared_state(&filename, &source) {
            Ok(typed_file) => {
                debug!(
                    "  ✓ Successfully compiled and registered: {}",
                    qualified_path
                );
                // Store typedef files so they're included in HIR conversion
                if !typed_file.type_aliases.is_empty() {
                    trace!(
                        "    (contains {} type aliases)",
                        typed_file.type_aliases.len()
                    );
                }

                // Check if any type aliases have Placeholder targets that need to be loaded
                // This handles cases like `typedef Bytes = rayzor.Bytes` where rayzor.Bytes hasn't been loaded yet
                let mut placeholder_targets = Vec::new();
                {
                    let type_table = self.type_table.borrow();
                    for alias in &typed_file.type_aliases {
                        if let Some(target_info) = type_table.get(alias.target_type) {
                            if let crate::tast::TypeKind::Placeholder { name } = &target_info.kind {
                                if let Some(placeholder_name) = self.string_interner.get(*name) {
                                    trace!(
                                        "    Found typedef with Placeholder target: {}",
                                        placeholder_name
                                    );
                                    placeholder_targets.push(placeholder_name.to_string());
                                }
                            }
                        }
                    }
                }

                // If we found Placeholder targets, try to load them and retry
                if !placeholder_targets.is_empty() {
                    let mut any_loaded = false;
                    for target in &placeholder_targets {
                        if let Ok(_) = self.load_import_file_recursive(target, depth + 1) {
                            debug!("    ✓ Loaded typedef target: {}", target);
                            any_loaded = true;
                        }
                    }

                    if any_loaded {
                        // Retry compilation after loading typedef targets
                        debug!(
                            "  Retrying compilation of {} after loading typedef targets...",
                            qualified_path
                        );
                        match self.compile_file_with_shared_state(&filename, &source) {
                            Ok(recompiled_file) => {
                                self.loaded_stdlib_typed_files.push(recompiled_file);
                                return Ok(());
                            }
                            Err(_) => {
                                // Fall through and push the original typed_file
                            }
                        }
                    }
                }

                self.loaded_stdlib_typed_files.push(typed_file);
                Ok(())
            }
            Err(errors) => {
                // Extract UnresolvedType errors and try to load those dependencies
                let mut missing_types = Vec::new();
                for error in &errors {
                    if let Some(type_name) =
                        Self::extract_unresolved_type_from_error(&error.message)
                    {
                        // Skip generic type parameters and built-in typedefs
                        if !Self::is_generic_type_parameter(&type_name)
                            && !self.failed_type_loads.contains(&type_name)
                        {
                            missing_types.push(type_name);
                        }
                    }
                }

                // If we found missing types, try to load them recursively
                if !missing_types.is_empty() {
                    debug!(
                        "  Detected {} missing dependencies for {}: {:?}",
                        missing_types.len(),
                        qualified_path,
                        missing_types
                    );

                    let mut load_success = false;
                    for missing_type in &missing_types {
                        // Check if this looks like a field reference (e.g., "haxe.SysTools.winMetaCharacters")
                        // If so, extract just the class part (e.g., "haxe.SysTools")
                        let type_to_load = if let Some(last_dot) = missing_type.rfind('.') {
                            let after_dot = &missing_type[last_dot + 1..];
                            // If the part after the last dot starts with lowercase, it's likely a field
                            if after_dot
                                .chars()
                                .next()
                                .map(|c| c.is_lowercase())
                                .unwrap_or(false)
                            {
                                &missing_type[..last_dot]
                            } else {
                                missing_type.as_str()
                            }
                        } else {
                            missing_type.as_str()
                        };

                        // Try loading with the (possibly adjusted) name first
                        let loaded = if let Ok(_) =
                            self.load_import_file_recursive(type_to_load, depth + 1)
                        {
                            debug!("    ✓ Loaded dependency: {}", type_to_load);
                            true
                        } else if !type_to_load.contains('.') {
                            // If unqualified name failed, try with common stdlib packages
                            let prefixes = vec!["haxe.exceptions.", "haxe.io.", "haxe.ds."];
                            let mut prefix_loaded = false;
                            for prefix in prefixes {
                                let qualified = format!("{}{}", prefix, type_to_load);
                                if let Ok(_) =
                                    self.load_import_file_recursive(&qualified, depth + 1)
                                {
                                    debug!(
                                        "    ✓ Loaded dependency: {} (as {})",
                                        type_to_load, qualified
                                    );
                                    prefix_loaded = true;
                                    break;
                                }
                            }
                            prefix_loaded
                        } else {
                            false
                        };

                        if loaded {
                            load_success = true;
                        } else {
                            debug!("    ✗ Could not load dependency: {}", missing_type);
                            self.failed_type_loads.insert(missing_type.clone());
                        }
                    }

                    // If we successfully loaded at least one dependency, retry compilation
                    if load_success {
                        debug!(
                            "  Retrying compilation of {} after loading dependencies...",
                            qualified_path
                        );
                        match self.compile_file_with_shared_state(&filename, &source) {
                            Ok(typed_file) => {
                                // Store typedef files so they're included in HIR conversion
                                if !typed_file.type_aliases.is_empty() {
                                    trace!(
                                        "    (contains {} type aliases after retry)",
                                        typed_file.type_aliases.len()
                                    );
                                }

                                // Check if any type aliases have Placeholder targets that need to be loaded
                                // This handles cases like `typedef Bytes = rayzor.Bytes` where rayzor.Bytes hasn't been loaded yet
                                let mut placeholder_targets = Vec::new();
                                {
                                    let type_table = self.type_table.borrow();
                                    for alias in &typed_file.type_aliases {
                                        if let Some(target_info) = type_table.get(alias.target_type)
                                        {
                                            if let crate::tast::TypeKind::Placeholder { name } =
                                                &target_info.kind
                                            {
                                                if let Some(placeholder_name) =
                                                    self.string_interner.get(*name)
                                                {
                                                    trace!("    Found typedef with Placeholder target (after deps): {}", placeholder_name);
                                                    placeholder_targets
                                                        .push(placeholder_name.to_string());
                                                }
                                            }
                                        }
                                    }
                                }

                                // If we found Placeholder targets, try to load them and retry again
                                if !placeholder_targets.is_empty() {
                                    let mut any_loaded = false;
                                    for target in &placeholder_targets {
                                        if let Ok(_) =
                                            self.load_import_file_recursive(target, depth + 1)
                                        {
                                            debug!(
                                                "    ✓ Loaded typedef target (after deps): {}",
                                                target
                                            );
                                            any_loaded = true;
                                        }
                                    }

                                    if any_loaded {
                                        // Retry compilation after loading typedef targets
                                        debug!("  Retrying compilation of {} after loading typedef targets...", qualified_path);
                                        match self
                                            .compile_file_with_shared_state(&filename, &source)
                                        {
                                            Ok(recompiled_file) => {
                                                self.loaded_stdlib_typed_files
                                                    .push(recompiled_file);
                                                return Ok(());
                                            }
                                            Err(_) => {
                                                // Fall through and push the original typed_file
                                            }
                                        }
                                    }
                                }

                                self.loaded_stdlib_typed_files.push(typed_file);
                                return Ok(());
                            }
                            Err(errors) => {
                                // Check if any errors are UnresolvedType that we can try to load
                                let mut additional_missing = Vec::new();
                                for error in &errors {
                                    if let Some(type_name) =
                                        Self::extract_unresolved_type_from_error(&error.message)
                                    {
                                        if !Self::is_generic_type_parameter(&type_name)
                                            && !self.failed_type_loads.contains(&type_name)
                                        {
                                            additional_missing.push(type_name);
                                        }
                                    }
                                }

                                if !additional_missing.is_empty() {
                                    let mut loaded_any = false;
                                    for missing in &additional_missing {
                                        if let Ok(_) =
                                            self.load_import_file_recursive(missing, depth + 1)
                                        {
                                            debug!(
                                                "    ✓ Loaded additional dependency: {}",
                                                missing
                                            );
                                            loaded_any = true;
                                        }
                                    }

                                    if loaded_any {
                                        // Try one more time
                                        debug!("  Retrying compilation of {} after loading additional dependencies...", qualified_path);
                                        match self
                                            .compile_file_with_shared_state(&filename, &source)
                                        {
                                            Ok(final_file) => {
                                                self.loaded_stdlib_typed_files.push(final_file);
                                                return Ok(());
                                            }
                                            Err(final_errors) => {
                                                let error_msgs: Vec<String> = final_errors
                                                    .iter()
                                                    .map(|e| e.message.clone())
                                                    .collect();
                                                return Err(format!("Errors compiling {} (after loading additional dependencies): {}", filename, error_msgs.join(", ")));
                                            }
                                        }
                                    }
                                }

                                let error_msgs: Vec<String> =
                                    errors.iter().map(|e| e.message.clone()).collect();
                                return Err(format!(
                                    "Errors compiling {} (after loading dependencies): {}",
                                    filename,
                                    error_msgs.join(", ")
                                ));
                            }
                        }
                    }
                }

                // No missing types found or couldn't load them - return original error
                let error_msgs: Vec<String> = errors.iter().map(|e| e.message.clone()).collect();
                Err(format!(
                    "Errors compiling {}: {}",
                    filename,
                    error_msgs.join(", ")
                ))
            }
        }
    }


    /// Extract type name from UnresolvedType error messages
    /// Returns Some(type_name) if this is an UnresolvedType error, None otherwise
    pub(crate) fn extract_unresolved_type_from_error(error_msg: &str) -> Option<String> {
        // Match pattern: "UnresolvedType { type_name: \"SomeType\", ..."
        if let Some(type_name_start) = error_msg.find("type_name: \"") {
            let after_marker = &error_msg[type_name_start + 12..]; // 12 = length of 'type_name: "'
            if let Some(end) = after_marker.find('"') {
                return Some(after_marker[..end].to_string());
            }
        }
        // Match pattern: "Cannot find type 'SomeType'"
        if let Some(start) = error_msg.find("Cannot find type '") {
            let after_marker = &error_msg[start + 18..]; // 18 = length of "Cannot find type '"
            if let Some(end) = after_marker.find('\'') {
                return Some(after_marker[..end].to_string());
            }
        }
        None
    }


    /// Check if a type name looks like a generic type parameter
    /// Returns true for single letters (T, K, V) or common parameter patterns
    pub(crate) fn is_generic_type_parameter(type_name: &str) -> bool {
        // Single uppercase letter
        if type_name.len() == 1
            && type_name
                .chars()
                .next()
                .map(|c| c.is_ascii_uppercase())
                .unwrap_or(false)
        {
            return true;
        }
        // Common generic parameter patterns and StdTypes prelude names.
        if Self::is_stdtypes_prelude_type_name(type_name) {
            return true;
        }
        matches!(type_name, "Key" | "Value" | "Item" | "Element")
    }


    pub(crate) fn is_stdtypes_prelude_type_name(type_name: &str) -> bool {
        is_stdtypes_ambient_name(type_name)
    }


    pub(crate) fn is_bare_stdtypes_prelude_dependency(dep: &str) -> bool {
        !dep.contains('.') && Self::is_stdtypes_prelude_type_name(dep)
    }


    /// Every `import.hx` that applies to this compilation, outermost first.
    ///
    /// Haxe honours an import.hx at a class-path root and applies it to every
    /// module at or below it. Three roots matter here:
    ///
    ///   1. the compiler's own stdlib, so rayzor can ship defaults with it;
    ///   2. the project root, the working directory the build was invoked from;
    ///   3. each user file's class-path root down to its own directory.
    ///
    /// The package declaration bounds (3) exactly: a file in `package cases`
    /// sits one directory below its class-path root, so that chain is
    /// package-depth + 1 long. Walking past the root would adopt an import.hx
    /// belonging to an unrelated tree.
    pub fn discover_import_hx_files(&self) -> Vec<PathBuf> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        let mut push = |d: PathBuf, dirs: &mut Vec<PathBuf>| {
            if d.is_dir() && !dirs.contains(&d) {
                dirs.push(d);
            }
        };

        for p in CompilationConfig::discover_stdlib_paths() {
            push(p, &mut dirs);
        }
        if let Ok(cwd) = std::env::current_dir() {
            push(cwd, &mut dirs);
        }
        for file in &self.user_files {
            let Some(parent) = PathBuf::from(&file.filename)
                .parent()
                .map(|p| p.to_path_buf())
            else {
                continue;
            };
            let dir = std::fs::canonicalize(&parent).unwrap_or(parent);
            let depth = file.package.as_ref().map(|p| p.path.len()).unwrap_or(0);
            let mut chain = Vec::new();
            let mut d = dir;
            for _ in 0..=depth {
                chain.push(d.clone());
                match d.parent() {
                    Some(up) => d = up.to_path_buf(),
                    None => break,
                }
            }
            chain.reverse();
            for c in chain {
                push(c, &mut dirs);
            }
        }

        dirs.into_iter()
            .map(|d| d.join("import.hx"))
            .filter(|p| p.is_file())
            .collect()
    }


    /// The types an `import.hx` names, so they can be loaded.
    ///
    /// An import.hx exists to name types nothing else in the program mentions,
    /// so those types are invisible to the ordinary import scan and never get
    /// compiled -- which is why a name an import.hx provides resolved to
    /// nothing even once the import itself was registered.
    ///
    /// The file is parsed for its import paths and NOTHING else. Compiling it
    /// as a user module instead installs a trap stub over the real `main` of
    /// whatever module sits beside it: it declares no types, and putting an
    /// empty module through the shared-state path renumbers the functions
    /// around it.
    pub(crate) fn import_hx_type_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for path in self.discover_import_hx_files() {
            let Ok(source) = std::fs::read_to_string(&path) else {
                continue;
            };
            let Ok(file) = self.parse_file(path.to_str().unwrap_or("import.hx"), &source) else {
                continue;
            };
            for import in &file.imports {
                if !import.path.is_empty() {
                    names.push(import.path.join("."));
                }
            }
            for using in &file.using {
                if !using.path.is_empty() {
                    names.push(using.path.join("."));
                }
            }
        }
        names
    }


    /// Load global import.hx files
    /// These are processed AFTER stdlib but BEFORE user files
    /// They provide global imports available to all user code
    pub fn load_global_imports(&mut self) -> Result<(), String> {
        use std::fs;

        for import_path in &self.config.global_import_hx_files.clone() {
            let source = fs::read_to_string(import_path)
                .map_err(|e| format!("Failed to read import.hx at {:?}: {}", import_path, e))?;

            let haxe_file = self
                .parse_file(import_path.to_str().unwrap_or("import.hx"), &source)
                .map_err(|e| format!("Parse error in {:?}: {}", import_path, e))?;

            self.import_hx_files.push(haxe_file);
        }

        Ok(())
    }


    /// Resolve an import path to a filesystem path
    /// For example: "com.example.model.User" -> "src/com/example/model/User.hx"
    ///
    /// # Arguments
    /// * `import_path` - The import path (e.g., "com.example.model.User")
    /// * `source_paths` - Directories to search for source files (e.g., ["src", "lib"])
    pub fn resolve_import_path(
        &self,
        import_path: &str,
        source_paths: &[PathBuf],
    ) -> Option<PathBuf> {
        // Convert import path to filesystem path
        // "com.example.model.User" -> "com/example/model/User.hx"
        let file_path = import_path.replace('.', "/") + ".hx";

        // Search in each source path
        for source_path in source_paths {
            let full_path = source_path.join(&file_path);
            if full_path.exists() {
                return Some(full_path);
            }
        }

        None
    }


    /// Add a file by import path (e.g., "com.example.model.User")
    /// This automatically searches source paths to find the file
    ///
    /// # Arguments
    /// * `import_path` - The import path
    /// * `source_paths` - Directories to search for source files
    pub fn add_file_by_import(
        &mut self,
        import_path: &str,
        source_paths: &[PathBuf],
    ) -> Result<(), String> {
        let path = self
            .resolve_import_path(import_path, source_paths)
            .ok_or_else(|| format!("Could not resolve import: {}", import_path))?;

        self.add_file_from_path(&path)
    }


    /// Where an unqualified type could live, reading outwards through the
    /// enclosing packages.
    ///
    /// Haxe resolves a bare type name against the enclosing packages, so a
    /// module in `unit.issues` reaches `unit.Test` as plain `Test` and
    /// `unit.HelperMacros` as plain `HelperMacros` — nothing in the file says
    /// which file to load. Supertypes are collected here from the declaration
    /// heads; `bare_names` carries the ones used in expression position, which
    /// the caller already gathered. Each is a candidate rather than a claim:
    /// `load_imports_efficiently` keeps the ones that resolve to a file and
    /// ignores the rest, so a guess that is wrong costs a failed path lookup.
    pub(crate) fn enclosing_package_candidates(
        ast: &parser::haxe_ast::HaxeFile,
        bare_names: &[String],
    ) -> Vec<String> {
        use parser::haxe_ast::{Type, TypeDeclaration};

        let Some(package) = ast.package.as_ref() else {
            return Vec::new();
        };
        if package.path.is_empty() {
            return Vec::new();
        }

        // Only a bare name is ambiguous; a qualified one already says where to look.
        let bare_name = |ty: &Type| -> Option<String> {
            match ty {
                Type::Path { path, .. } if path.package.is_empty() && path.sub.is_none() => {
                    Some(path.name.clone())
                }
                _ => None,
            }
        };

        let mut names: Vec<String> = Vec::new();
        for decl in &ast.declarations {
            match decl {
                TypeDeclaration::Class(class) => {
                    names.extend(class.extends.as_ref().and_then(bare_name));
                    names.extend(class.implements.iter().filter_map(bare_name));
                }
                TypeDeclaration::Interface(iface) => {
                    names.extend(iface.extends.iter().filter_map(bare_name));
                }
                _ => {}
            }
        }
        // Bare capitalized names used in expression position — `HelperMacros`
        // in `HelperMacros.typeString(a)` is a type reference, and by Haxe
        // convention only a type starts uppercase there.
        names.extend(
            bare_names
                .iter()
                .filter(|n| !n.contains('.') && n.chars().next().is_some_and(|c| c.is_uppercase()))
                .cloned(),
        );
        names.sort();
        names.dedup();
        if names.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        for name in names {
            // Nearest enclosing package first, so a closer definition wins.
            for depth in (1..package.path.len()).rev() {
                candidates.push(format!("{}.{}", package.path[..depth].join("."), name));
            }
        }
        candidates
    }


    /// Analyze dependencies and get compilation order
    ///
    /// This builds a dependency graph from all user files and determines
    /// the correct compilation order. It also detects circular dependencies.
    ///
    /// Returns (compilation_order, circular_dependencies)
    pub fn analyze_dependencies(&self) -> Result<DependencyAnalysis, Vec<CompilationError>> {
        if self.user_files.is_empty() {
            return Ok(DependencyAnalysis {
                compilation_order: Vec::new(),
                circular_dependencies: Vec::new(),
            });
        }

        // Build dependency graph
        let graph = DependencyGraph::from_files(&self.user_files);

        // Analyze
        let analysis = graph.analyze();

        // Report circular dependencies as warnings (not errors)
        if !analysis.circular_dependencies.is_empty() {
            debug!("⚠️  Warning: Circular dependencies detected!");
            for (i, cycle) in analysis.circular_dependencies.iter().enumerate() {
                debug!("\nCycle #{}:", i + 1);
                debug!("{}", cycle.format_error());
            }
            debug!("\nCompilation will proceed with best-effort ordering.\n");
        }

        Ok(analysis)
    }
}
