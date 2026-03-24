//! Haxe-to-WGSL shader transpiler.
//!
//! Transpiles `@:shader` classes to valid WGSL source code at compile time.
//! Called during MIR lowering when `MyShader.wgsl()` is encountered.
//!
//! Automatically detects `@:gpuStruct` types used in shader method parameters
//! and generates the corresponding WGSL struct definitions with `@location(N)`.

use crate::ir::hir;
use crate::tast::node::{
    BinaryOperator, LiteralValue, TypedClass, TypedExpression, TypedExpressionKind, TypedField,
    TypedFunction, TypedStatement, UnaryOperator,
};
use crate::tast::symbols::{SymbolFlags, SymbolKind};
use crate::tast::{StringInterner, SymbolId, SymbolTable, TypeId, TypeKind, TypeTable};
use std::collections::BTreeSet;
use std::fmt::Write;

/// Transpile a @:shader class from HIR (available during MIR lowering).
/// Generates WGSL struct definitions for @:gpuStruct parameters and
/// entry point functions for methods.
pub fn transpile_shader_from_hir(
    hir_class: &hir::HirClass,
    symbol_table: &SymbolTable,
    type_table: &TypeTable,
    interner: &StringInterner,
    hir_types: &indexmap::IndexMap<crate::tast::TypeId, hir::HirTypeDecl>,
) -> Result<String, String> {
    let mut ctx = WgslCtx {
        st: symbol_table,
        tt: type_table,
        si: interner,
        hir_types,
        out: String::new(),
        emitted_structs: BTreeSet::new(),
    };

    // 1. Emit struct dependencies from method params/returns
    for method in &hir_class.methods {
        for param in &method.function.params {
            ctx.maybe_emit_struct(param.ty, true)?;
        }
        ctx.maybe_emit_struct(method.function.return_type, false)?;
    }

    // 2. Emit uniform bindings from class fields
    for (binding_idx, field) in hir_class.fields.iter().enumerate() {
        let name = interner.get(field.name).unwrap_or("u");
        let wtype = ctx.type_to_wgsl(field.ty);
        writeln!(
            ctx.out,
            "@group(0) @binding({}) var<uniform> {}: {};",
            binding_idx, name, wtype
        )
        .unwrap();
    }
    if !hir_class.fields.is_empty() {
        ctx.out.push('\n');
    }

    // 3. Emit entry point functions (skip synthetic wgsl())
    for method in &hir_class.methods {
        let name = interner.get(method.function.name).unwrap_or("fn_unknown");
        if name == "wgsl" {
            continue; // Skip the synthetic method itself
        }

        let stage = if name == "vertex" || name.starts_with("vs_") {
            Some("vertex")
        } else if name == "fragment" || name.starts_with("fs_") {
            Some("fragment")
        } else if name == "compute" || name.starts_with("cs_") {
            Some("compute")
        } else {
            None
        };

        if let Some(s) = stage {
            writeln!(ctx.out, "@{}", s).unwrap();
        }

        write!(ctx.out, "fn {}(", name).unwrap();
        for (i, param) in method.function.params.iter().enumerate() {
            if i > 0 {
                ctx.out.push_str(", ");
            }
            let pname = interner.get(param.name).unwrap_or("p");
            let ptype = ctx.type_to_wgsl(param.ty);
            // Auto-detect builtin parameters by name convention
            // Auto-detect builtin parameters by name — use correct WGSL types
            let builtin = match pname {
                "vertexIndex" | "vertex_index" => Some(("vertex_index", "u32")),
                "instanceIndex" | "instance_index" => Some(("instance_index", "u32")),
                "frontFacing" | "front_facing" => Some(("front_facing", "bool")),
                "sampleIndex" | "sample_index" => Some(("sample_index", "u32")),
                _ => None,
            };
            if let Some((b, btype)) = builtin {
                write!(ctx.out, "@builtin({}) {}: {}", b, pname, btype).unwrap();
            } else if stage == Some("vertex") && ctx.is_gpu_struct(param.ty) {
                // Vertex input struct — no annotation needed (struct fields have @location)
                write!(ctx.out, "{}: {}", pname, ptype).unwrap();
            } else if stage == Some("fragment") {
                // Fragment input from vertex output — no annotation needed
                write!(ctx.out, "{}: {}", pname, ptype).unwrap();
            } else {
                write!(ctx.out, "{}: {}", pname, ptype).unwrap();
            }
        }
        ctx.out.push_str(") -> ");

        let ret = ctx.type_to_wgsl(method.function.return_type);
        if ctx.is_gpu_struct(method.function.return_type) {
            write!(ctx.out, "{}", ret).unwrap();
        } else if stage == Some("vertex") && ret == "vec4f" {
            write!(ctx.out, "@builtin(position) {}", ret).unwrap();
        } else if stage == Some("fragment") {
            write!(ctx.out, "@location(0) {}", ret).unwrap();
        } else {
            write!(ctx.out, "{}", ret).unwrap();
        }

        ctx.out.push_str(" {\n");

        // Transpile HIR function body (trailing expression becomes return)
        if let Some(ref body) = method.function.body {
            ctx.emit_hir_body_as_return(body, 1)?;
        }
        // Also check: body might be in the trailing expr
        if let Some(ref body) = method.function.body {
            if body.statements.is_empty() {
                if let Some(ref trail) = body.expr {
                    // The entire function body is a single expression (common in HIR)
                    if let hir::HirExprKind::Block(inner) = &trail.kind {
                        for stmt in &inner.statements {
                            ctx.emit_hir_stmt(stmt, 1)?;
                        }
                        if let Some(ref inner_trail) = inner.expr {
                            let ind = ctx.ind(1);
                            let e = ctx.hir_expr_to_string(inner_trail)?;
                            writeln!(ctx.out, "{}return {};", ind, e).unwrap();
                        }
                    }
                }
            }
        }

        ctx.out.push_str("}\n\n");
    }

    Ok(ctx.out)
}

/// Transpile a @:shader class to WGSL source code (from TAST).
pub fn transpile_shader_class(
    class: &TypedClass,
    symbol_table: &SymbolTable,
    type_table: &TypeTable,
    interner: &StringInterner,
) -> Result<String, String> {
    let mut ctx = WgslCtx {
        st: symbol_table,
        tt: type_table,
        si: interner,
        hir_types: &indexmap::IndexMap::new(),
        out: String::new(),
        emitted_structs: BTreeSet::new(),
    };

    // 1. Emit struct dependencies from method params/returns
    for method in &class.methods {
        for param in &method.parameters {
            ctx.maybe_emit_struct(param.param_type, true)?;
        }
        ctx.maybe_emit_struct(method.return_type, false)?;
    }

    // 2. Emit uniform bindings from class fields
    for (binding_idx, field) in class.fields.iter().enumerate() {
        let name = interner.get(field.name).unwrap_or("u");
        let wtype = ctx.type_to_wgsl(field.field_type);
        writeln!(
            ctx.out,
            "@group(0) @binding({}) var<uniform> {}: {};",
            binding_idx, name, wtype
        )
        .unwrap();
    }
    if !class.fields.is_empty() {
        ctx.out.push('\n');
    }

    // 3. Emit entry point functions
    for method in &class.methods {
        let name = interner.get(method.name).unwrap_or("fn_unknown");

        let stage = if name == "vertex" || name.starts_with("vs_") {
            Some("vertex")
        } else if name == "fragment" || name.starts_with("fs_") {
            Some("fragment")
        } else if name == "compute" || name.starts_with("cs_") {
            Some("compute")
        } else {
            None
        };

        if let Some(s) = stage {
            writeln!(ctx.out, "@{}", s).unwrap();
        }

        write!(ctx.out, "fn {}(", name).unwrap();
        for (i, param) in method.parameters.iter().enumerate() {
            if i > 0 {
                ctx.out.push_str(", ");
            }
            let pname = interner.get(param.name).unwrap_or("p");
            let ptype = ctx.type_to_wgsl(param.param_type);
            write!(ctx.out, "{}: {}", pname, ptype).unwrap();
        }
        ctx.out.push_str(") -> ");

        let ret = ctx.type_to_wgsl(method.return_type);
        if ctx.is_gpu_struct(method.return_type) {
            write!(ctx.out, "{}", ret).unwrap();
        } else if stage == Some("vertex") && ret == "vec4f" {
            write!(ctx.out, "@builtin(position) {}", ret).unwrap();
        } else if stage == Some("fragment") {
            write!(ctx.out, "@location(0) {}", ret).unwrap();
        } else {
            write!(ctx.out, "{}", ret).unwrap();
        }

        ctx.out.push_str(" {\n");

        for stmt in &method.body {
            ctx.emit_stmt(stmt, 1)?;
        }

        ctx.out.push_str("}\n\n");
    }

    Ok(ctx.out)
}

// ---------------------------------------------------------------------------

struct WgslCtx<'a> {
    st: &'a SymbolTable,
    tt: &'a TypeTable,
    si: &'a StringInterner,
    hir_types: &'a indexmap::IndexMap<crate::tast::TypeId, hir::HirTypeDecl>,
    out: String,
    emitted_structs: BTreeSet<String>,
}

impl<'a> WgslCtx<'a> {
    fn ind(&self, n: usize) -> String {
        "    ".repeat(n)
    }

    fn sym_name(&self, id: SymbolId) -> &str {
        self.st
            .get_symbol(id)
            .and_then(|s| self.si.get(s.name))
            .unwrap_or("_")
    }

    fn type_to_wgsl(&self, id: TypeId) -> String {
        let ti = match self.tt.get(id) {
            Some(t) => t,
            None => return "f32".into(),
        };
        match &ti.kind {
            TypeKind::Void => "void".into(),
            TypeKind::Bool => "bool".into(),
            TypeKind::Int => "i32".into(),
            TypeKind::Float => "f32".into(),
            TypeKind::Class { symbol_id, .. } => {
                let n = self.sym_name(*symbol_id);
                match n {
                    "Vec2" => "vec2f",
                    "Vec3" => "vec3f",
                    "Vec4" => "vec4f",
                    "Mat4" => "mat4x4f",
                    "Mat3" => "mat3x3f",
                    _ => n,
                }
                .to_string()
            }
            TypeKind::Array { element_type, .. } => {
                format!("array<{}>", self.type_to_wgsl(*element_type))
            }
            _ => "f32".into(),
        }
    }

    fn is_gpu_struct(&self, id: TypeId) -> bool {
        if let Some(ti) = self.tt.get(id) {
            if let TypeKind::Class { symbol_id, .. } = &ti.kind {
                return self
                    .st
                    .get_symbol(*symbol_id)
                    .map(|s| s.flags.is_gpu_struct())
                    .unwrap_or(false);
            }
        }
        false
    }

    fn maybe_emit_struct(&mut self, type_id: TypeId, is_input: bool) -> Result<(), String> {
        let ti = match self.tt.get(type_id) {
            Some(t) => t,
            None => return Ok(()),
        };
        let (sym_id, name) = match &ti.kind {
            TypeKind::Class { symbol_id, .. } => {
                let s = match self.st.get_symbol(*symbol_id) {
                    Some(s) if s.flags.is_gpu_struct() => s,
                    _ => return Ok(()),
                };
                let n = self.si.get(s.name).unwrap_or("Struct");
                if matches!(n, "Vec2" | "Vec3" | "Vec4" | "Mat4" | "Mat3") {
                    return Ok(());
                }
                (*symbol_id, n.to_string())
            }
            _ => return Ok(()),
        };

        if self.emitted_structs.contains(&name) {
            return Ok(());
        }
        self.emitted_structs.insert(name.clone());

        writeln!(self.out, "struct {} {{", name).unwrap();
        let mut loc = 0u32;

        // Find fields by qualified name containing the class name,
        // or by matching the class scope_id OR definition_location file.
        // Get fields from the HIR class by matching class NAME (symbol_id may differ)
        let mut fields: Vec<(String, String)> = Vec::new();
        for (_tid, decl) in self.hir_types {
            if let hir::HirTypeDecl::Class(c) = decl {
                let cname = self.si.get(c.name).unwrap_or("");
                if cname == name {
                    for f in &c.fields {
                        let fname = self.si.get(f.name).unwrap_or("f");
                        if fname.starts_with("__") {
                            continue;
                        }
                        let ftype = self.type_to_wgsl(f.ty);
                        fields.push((fname.to_string(), ftype));
                    }
                    break;
                }
            }
        }

        for (fname, ftype) in &fields {
            let ann = if !is_input && fname == "position" && ftype == "vec4f" {
                "@builtin(position) ".to_string()
            } else {
                let a = format!("@location({}) ", loc);
                loc += 1;
                a
            };
            writeln!(self.out, "    {}{}: {},", ann, fname, ftype).unwrap();
        }

        self.out.push_str("}\n\n");
        Ok(())
    }

    fn emit_stmt(&mut self, stmt: &TypedStatement, depth: usize) -> Result<(), String> {
        let ind = self.ind(depth);
        match stmt {
            TypedStatement::Expression { expression, .. } => {
                let e = self.emit_expr(expression)?;
                writeln!(self.out, "{}{};", ind, e).unwrap();
            }
            TypedStatement::VarDeclaration {
                symbol_id,
                initializer,
                ..
            } => {
                let name = self.sym_name(*symbol_id).to_string();
                let ty = self
                    .st
                    .get_symbol(*symbol_id)
                    .map(|s| self.type_to_wgsl(s.type_id))
                    .unwrap_or_else(|| "f32".into());
                if let Some(init) = initializer {
                    let val = self.emit_expr(init)?;
                    writeln!(self.out, "{}var {}: {} = {};", ind, name, ty, val).unwrap();
                } else {
                    writeln!(self.out, "{}var {}: {};", ind, name, ty).unwrap();
                }
            }
            TypedStatement::Assignment { target, value, .. } => {
                let t = self.emit_expr(target)?;
                let v = self.emit_expr(value)?;
                writeln!(self.out, "{}{} = {};", ind, t, v).unwrap();
            }
            TypedStatement::Return { value, .. } => {
                if let Some(val) = value {
                    let v = self.emit_expr(val)?;
                    writeln!(self.out, "{}return {};", ind, v).unwrap();
                } else {
                    writeln!(self.out, "{}return;", ind).unwrap();
                }
            }
            TypedStatement::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let c = self.emit_expr(condition)?;
                writeln!(self.out, "{}if ({}) {{", ind, c).unwrap();
                self.emit_stmt(then_branch, depth + 1)?;
                if let Some(eb) = else_branch {
                    writeln!(self.out, "{}}} else {{", ind).unwrap();
                    self.emit_stmt(eb, depth + 1)?;
                }
                writeln!(self.out, "{}}}", ind).unwrap();
            }
            TypedStatement::Block { statements, .. } => {
                for s in statements {
                    self.emit_stmt(s, depth)?;
                }
            }
            _ => {
                writeln!(self.out, "{}/* unsupported statement */", ind).unwrap();
            }
        }
        Ok(())
    }

    fn emit_expr(&self, expr: &TypedExpression) -> Result<String, String> {
        match &expr.kind {
            TypedExpressionKind::Literal { value } => Ok(self.emit_lit(value)),
            TypedExpressionKind::Variable { symbol_id, .. } => {
                Ok(self.sym_name(*symbol_id).to_string())
            }
            TypedExpressionKind::BinaryOp {
                left,
                operator,
                right,
                ..
            } => {
                let l = self.emit_expr(left)?;
                let r = self.emit_expr(right)?;
                Ok(format!("({} {} {})", l, self.binop(operator), r))
            }
            TypedExpressionKind::UnaryOp {
                operator, operand, ..
            } => {
                let e = self.emit_expr(operand)?;
                Ok(format!("({}{})", self.unop(operator), e))
            }
            TypedExpressionKind::FieldAccess {
                object,
                field_symbol,
                ..
            } => {
                let o = self.emit_expr(object)?;
                let f = self.sym_name(*field_symbol);
                Ok(format!("{}.{}", o, f))
            }
            TypedExpressionKind::MethodCall {
                receiver,
                method_symbol,
                arguments,
                ..
            } => {
                let recv = self.emit_expr(receiver)?;
                let mname = self.sym_name(*method_symbol);
                let args = self.emit_args(arguments)?;
                // Map methods to WGSL builtins
                match mname {
                    "dot" => Ok(format!("dot({}, {})", recv, args)),
                    "cross" => Ok(format!("cross({}, {})", recv, args)),
                    "normalize" => Ok(format!("normalize({})", recv)),
                    "length" => Ok(format!("length({})", recv)),
                    "scale" => Ok(format!("({} * {})", recv, args)),
                    _ => Ok(format!("{}.{}", recv, mname)),
                }
            }
            TypedExpressionKind::StaticMethodCall {
                class_symbol,
                method_symbol,
                arguments,
                ..
            } => {
                let cls = self.sym_name(*class_symbol);
                let meth = self.sym_name(*method_symbol);
                let args = self.emit_args(arguments)?;
                self.emit_static(cls, meth, &args)
            }
            TypedExpressionKind::FunctionCall {
                function,
                arguments,
                ..
            } => {
                let f = self.emit_expr(function)?;
                let args = self.emit_args(arguments)?;
                Ok(format!("{}({})", f, args))
            }
            TypedExpressionKind::New {
                class_type,
                arguments,
                ..
            } => {
                let t = self.type_to_wgsl(*class_type);
                let args = self.emit_args(arguments)?;
                Ok(format!("{}({})", t, args))
            }
            TypedExpressionKind::ArrayAccess { array, index, .. } => {
                let a = self.emit_expr(array)?;
                let i = self.emit_expr(index)?;
                Ok(format!("{}[{}]", a, i))
            }
            TypedExpressionKind::Cast {
                expression,
                target_type,
                ..
            } => {
                let e = self.emit_expr(expression)?;
                let t = self.type_to_wgsl(*target_type);
                Ok(format!("{}({})", t, e))
            }
            TypedExpressionKind::Null => Ok("0".into()),
            TypedExpressionKind::This { .. } => Ok("self".into()),
            TypedExpressionKind::Conditional {
                condition,
                then_expr,
                else_expr,
            } => {
                let c = self.emit_expr(condition)?;
                let t = self.emit_expr(then_expr)?;
                if let Some(e_expr) = else_expr {
                    let e = self.emit_expr(e_expr)?;
                    Ok(format!("select({}, {}, {})", e, t, c))
                } else {
                    Ok(format!("select(0, {}, {})", t, c))
                }
            }
            TypedExpressionKind::Return { value, .. } => {
                if let Some(v) = value {
                    let ve = self.emit_expr(v)?;
                    Ok(format!("return {}", ve))
                } else {
                    Ok("return".into())
                }
            }
            _ => Ok("/* unsupported */".into()),
        }
    }

    fn emit_lit(&self, lit: &LiteralValue) -> String {
        match lit {
            LiteralValue::Int(n) => n.to_string(),
            LiteralValue::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') {
                    format!("{}f", s)
                } else {
                    format!("{}.0f", s)
                }
            }
            LiteralValue::Bool(b) => b.to_string(),
            LiteralValue::String(s) => format!("/* \"{}\" */", s),
            _ => "0".into(),
        }
    }

    fn emit_args(&self, args: &[TypedExpression]) -> Result<String, String> {
        let strs: Result<Vec<String>, String> = args.iter().map(|a| self.emit_expr(a)).collect();
        Ok(strs?.join(", "))
    }

    fn emit_static(&self, cls: &str, meth: &str, args: &str) -> Result<String, String> {
        Ok(match (cls, meth) {
            ("Math", "sqrt") => format!("sqrt({})", args),
            ("Math", "sin") => format!("sin({})", args),
            ("Math", "cos") => format!("cos({})", args),
            ("Math", "tan") => format!("tan({})", args),
            ("Math", "abs") => format!("abs({})", args),
            ("Math", "min") => format!("min({})", args),
            ("Math", "max") => format!("max({})", args),
            ("Math", "pow") => format!("pow({})", args),
            ("Math", "floor") => format!("floor({})", args),
            ("Math", "ceil") => format!("ceil({})", args),
            ("Math", "exp") => format!("exp({})", args),
            ("Math", "log") => format!("log({})", args),
            ("ShaderMath", m) => format!("{}({})", m, args),
            ("Vec4", "fromVec3") => format!("vec4f({})", args),
            _ => format!("{}({})", meth, args),
        })
    }

    fn binop(&self, op: &BinaryOperator) -> &'static str {
        match op {
            BinaryOperator::Add => "+",
            BinaryOperator::Sub => "-",
            BinaryOperator::Mul => "*",
            BinaryOperator::Div => "/",
            BinaryOperator::Mod => "%",
            BinaryOperator::Eq => "==",
            BinaryOperator::Ne => "!=",
            BinaryOperator::Lt => "<",
            BinaryOperator::Le => "<=",
            BinaryOperator::Gt => ">",
            BinaryOperator::Ge => ">=",
            BinaryOperator::And => "&&",
            BinaryOperator::Or => "||",
            BinaryOperator::BitAnd => "&",
            BinaryOperator::BitOr => "|",
            BinaryOperator::BitXor => "^",
            BinaryOperator::Shl => "<<",
            BinaryOperator::Shr => ">>",
            _ => "/* op */",
        }
    }

    fn unop(&self, op: &UnaryOperator) -> &'static str {
        match op {
            UnaryOperator::Neg => "-",
            UnaryOperator::Not => "!",
            UnaryOperator::BitNot => "~",
            _ => "",
        }
    }

    // -----------------------------------------------------------------------
    // HIR expression/statement transpilation (used by transpile_shader_from_hir)
    // -----------------------------------------------------------------------

    fn emit_hir_stmt(&mut self, stmt: &hir::HirStatement, depth: usize) -> Result<(), String> {
        let ind = self.ind(depth);
        match stmt {
            hir::HirStatement::Expr(expr) => {
                match &expr.kind {
                    // Expand nested blocks inline
                    hir::HirExprKind::Block(block) => {
                        self.emit_hir_body(block, depth)?;
                    }
                    // If expression used as statement → emit WGSL if/else block
                    hir::HirExprKind::If {
                        condition,
                        then_expr,
                        else_expr,
                    } => {
                        let c = self.hir_expr_to_string(condition)?;
                        writeln!(self.out, "{}if ({}) {{", ind, c).unwrap();
                        if let hir::HirExprKind::Block(then_block) = &then_expr.kind {
                            self.emit_hir_body(then_block, depth + 1)?;
                        } else {
                            let t = self.hir_expr_to_string(then_expr)?;
                            writeln!(self.out, "{}    {};", ind, t).unwrap();
                        }
                        // Check else branch
                        if let hir::HirExprKind::Block(else_block) = &else_expr.kind {
                            if !else_block.statements.is_empty() || else_block.expr.is_some() {
                                writeln!(self.out, "{}}} else {{", ind).unwrap();
                                self.emit_hir_body(else_block, depth + 1)?;
                            }
                        } else if !matches!(else_expr.kind, hir::HirExprKind::Null) {
                            writeln!(self.out, "{}}} else {{", ind).unwrap();
                            let e = self.hir_expr_to_string(else_expr)?;
                            writeln!(self.out, "{}    {};", ind, e).unwrap();
                        }
                        writeln!(self.out, "{}}}", ind).unwrap();
                    }
                    _ => {
                        // Skip side-effect-free expression statements
                        // (dead code from HIR assignment trailing values)
                        if matches!(
                            expr.kind,
                            hir::HirExprKind::Variable { .. }
                                | hir::HirExprKind::Field { .. }
                                | hir::HirExprKind::Null
                        ) {
                            // Skip — assignment result value, not a real statement
                        } else {
                            let s = self.hir_expr_to_string(expr)?;
                            if !s.is_empty() {
                                writeln!(self.out, "{}{};", ind, s).unwrap();
                            }
                        }
                    }
                }
            }
            hir::HirStatement::Let {
                pattern,
                type_hint,
                init,
                ..
            } => {
                let name = match pattern {
                    hir::HirPattern::Variable { symbol, .. } => {
                        self.sym_name(*symbol).to_string()
                    }
                    _ => "_".to_string(),
                };
                let ty = type_hint
                    .map(|t| self.type_to_wgsl(t))
                    .unwrap_or_else(|| "f32".into());
                if let Some(init_expr) = init {
                    let val = self.hir_expr_to_string(init_expr)?;
                    writeln!(self.out, "{}var {}: {} = {};", ind, name, ty, val).unwrap();
                } else {
                    writeln!(self.out, "{}var {}: {};", ind, name, ty).unwrap();
                }
            }
            hir::HirStatement::Assign { lhs, rhs, .. } => {
                let t = self.hir_lvalue_to_string(lhs)?;
                let v = self.hir_expr_to_string(rhs)?;
                writeln!(self.out, "{}{} = {};", ind, t, v).unwrap();
            }
            hir::HirStatement::Return(value) => {
                if let Some(val) = value {
                    let v = self.hir_expr_to_string(val)?;
                    writeln!(self.out, "{}return {};", ind, v).unwrap();
                } else {
                    writeln!(self.out, "{}return;", ind).unwrap();
                }
            }
            hir::HirStatement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let c = self.hir_expr_to_string(condition)?;
                writeln!(self.out, "{}if ({}) {{", ind, c).unwrap();
                for s in &then_branch.statements {
                    self.emit_hir_stmt(s, depth + 1)?;
                }
                if let Some(eb) = else_branch {
                    writeln!(self.out, "{}}} else {{", ind).unwrap();
                    for s in &eb.statements {
                        self.emit_hir_stmt(s, depth + 1)?;
                    }
                }
                writeln!(self.out, "{}}}", ind).unwrap();
            }
            _ => {
                writeln!(self.out, "{}/* unsupported HIR statement */", ind).unwrap();
            }
        }
        Ok(())
    }

    fn hir_expr_to_string(&self, expr: &hir::HirExpr) -> Result<String, String> {
        use hir::HirExprKind;
        match &expr.kind {
            HirExprKind::Literal(lit) => Ok(self.hir_lit(lit)),
            HirExprKind::Variable { symbol, .. } => Ok(self.sym_name(*symbol).to_string()),
            HirExprKind::Field { object, field } => {
                let o = self.hir_expr_to_string(object)?;
                let f = self.sym_name(*field);
                Ok(format!("{}.{}", o, f))
            }
            HirExprKind::Index { object, index } => {
                let o = self.hir_expr_to_string(object)?;
                let i = self.hir_expr_to_string(index)?;
                Ok(format!("{}[{}]", o, i))
            }
            HirExprKind::Call { callee, args, .. } => {
                let f = self.hir_expr_to_string(callee)?;
                let arg_strs: Result<Vec<String>, String> =
                    args.iter().map(|a| self.hir_expr_to_string(a)).collect();
                Ok(format!("{}({})", f, arg_strs?.join(", ")))
            }
            HirExprKind::New { class_type, args, .. } => {
                let t = self.type_to_wgsl(*class_type);
                let arg_strs: Result<Vec<String>, String> =
                    args.iter().map(|a| self.hir_expr_to_string(a)).collect();
                Ok(format!("{}({})", t, arg_strs?.join(", ")))
            }
            HirExprKind::Binary { op, lhs, rhs } => {
                let l = self.hir_expr_to_string(lhs)?;
                let r = self.hir_expr_to_string(rhs)?;
                Ok(format!("({} {} {})", l, self.hir_binop(op), r))
            }
            HirExprKind::Unary { op, operand } => {
                let e = self.hir_expr_to_string(operand)?;
                Ok(format!("({}{})", self.hir_unop(op), e))
            }
            HirExprKind::Cast { expr, target, .. } => {
                let e = self.hir_expr_to_string(expr)?;
                let t = self.type_to_wgsl(*target);
                Ok(format!("{}({})", t, e))
            }
            HirExprKind::If {
                condition,
                then_expr,
                else_expr,
            } => {
                let c = self.hir_expr_to_string(condition)?;
                let t = self.hir_expr_to_string(then_expr)?;
                let e = self.hir_expr_to_string(else_expr)?;
                Ok(format!("select({}, {}, {})", e, t, c))
            }
            HirExprKind::Block(block) => self.hir_block_to_string(block),
            HirExprKind::This => Ok("self".into()),
            HirExprKind::Null => Ok("0".into()),
            _ => Ok("/* unsupported HIR expr */".into()),
        }
    }

    fn hir_block_to_string(&self, block: &hir::HirBlock) -> Result<String, String> {
        // For blocks used as expressions (e.g., if/else), return trailing expr
        if let Some(ref expr) = block.expr {
            // Recursively unwrap nested blocks
            if let hir::HirExprKind::Block(inner) = &expr.kind {
                return self.hir_block_to_string(inner);
            }
            return self.hir_expr_to_string(expr);
        }
        // Return last expression statement
        if let Some(last) = block.statements.last() {
            if let hir::HirStatement::Expr(e) = last {
                return self.hir_expr_to_string(e);
            }
            if let hir::HirStatement::Return(Some(e)) = last {
                return self.hir_expr_to_string(e);
            }
        }
        Ok("".into())
    }

    fn hir_lit(&self, lit: &hir::HirLiteral) -> String {
        match lit {
            hir::HirLiteral::Int(n) => n.to_string(),
            hir::HirLiteral::Float(f) => {
                let s = format!("{}", f);
                if s.contains('.') { format!("{}f", s) } else { format!("{}.0f", s) }
            }
            hir::HirLiteral::Bool(b) => b.to_string(),
            hir::HirLiteral::String(s) => {
                let val = self.si.get(*s).unwrap_or("");
                format!("/* \"{}\" */", val)
            }
            _ => "0".into(),
        }
    }

    fn hir_binop(&self, op: &hir::HirBinaryOp) -> &'static str {
        use hir::HirBinaryOp;
        match op {
            HirBinaryOp::Add => "+",
            HirBinaryOp::Sub => "-",
            HirBinaryOp::Mul => "*",
            HirBinaryOp::Div => "/",
            HirBinaryOp::Mod => "%",
            HirBinaryOp::Eq => "==",
            HirBinaryOp::Ne => "!=",
            HirBinaryOp::Lt => "<",
            HirBinaryOp::Le => "<=",
            HirBinaryOp::Gt => ">",
            HirBinaryOp::Ge => ">=",
            HirBinaryOp::And => "&&",
            HirBinaryOp::Or => "||",
            HirBinaryOp::BitAnd => "&",
            HirBinaryOp::BitOr => "|",
            HirBinaryOp::BitXor => "^",
            HirBinaryOp::Shl => "<<",
            HirBinaryOp::Shr => ">>",
            _ => "/* op */",
        }
    }

    /// Emit an HIR block body — handles nested Block expressions transparently.
    fn emit_hir_body(&mut self, body: &hir::HirBlock, depth: usize) -> Result<(), String> {
        self.emit_hir_body_inner(body, depth, false)
    }

    fn emit_hir_body_as_return(&mut self, body: &hir::HirBlock, depth: usize) -> Result<(), String> {
        self.emit_hir_body_inner(body, depth, true)
    }

    fn emit_hir_body_inner(
        &mut self,
        body: &hir::HirBlock,
        depth: usize,
        emit_return: bool,
    ) -> Result<(), String> {
        for stmt in &body.statements {
            self.emit_hir_stmt(stmt, depth)?;
        }
        if let Some(ref trailing) = body.expr {
            if let hir::HirExprKind::Block(inner) = &trailing.kind {
                self.emit_hir_body_inner(inner, depth, emit_return)?;
            } else if emit_return {
                let ind = self.ind(depth);
                let e = self.hir_expr_to_string(trailing)?;
                writeln!(self.out, "{}return {};", ind, e).unwrap();
            } else {
                // Skip trailing expressions that are just assignment results
                // (field access, variable ref — these are the value of the preceding assignment)
                let is_assignment_result = matches!(
                    trailing.kind,
                    hir::HirExprKind::Variable { .. }
                        | hir::HirExprKind::Field { .. }
                        | hir::HirExprKind::Null
                );
                if !is_assignment_result {
                    let ind = self.ind(depth);
                    let e = self.hir_expr_to_string(trailing)?;
                    if !e.is_empty() {
                        writeln!(self.out, "{}{};", ind, e).unwrap();
                    }
                }
            }
        }
        Ok(())
    }

    fn hir_lvalue_to_string(&self, lv: &hir::HirLValue) -> Result<String, String> {
        match lv {
            hir::HirLValue::Variable(sym) => Ok(self.sym_name(*sym).to_string()),
            hir::HirLValue::Field { object, field } => {
                let o = self.hir_expr_to_string(object)?;
                let f = self.sym_name(*field);
                Ok(format!("{}.{}", o, f))
            }
            hir::HirLValue::Index { object, index } => {
                let o = self.hir_expr_to_string(object)?;
                let i = self.hir_expr_to_string(index)?;
                Ok(format!("{}[{}]", o, i))
            }
        }
    }

    fn hir_unop(&self, op: &hir::HirUnaryOp) -> &'static str {
        use hir::HirUnaryOp;
        match op {
            HirUnaryOp::Neg => "-",
            HirUnaryOp::Not => "!",
            HirUnaryOp::BitNot => "~",
            _ => "",
        }
    }
}
