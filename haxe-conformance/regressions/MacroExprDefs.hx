import haxe.macro.Expr;
import haxe.macro.Context;

// haxe.macro.Expr both ways: a macro takes any expression apart by its
// ExprDef constructors, and an ExprDef it builds becomes that expression.
// A macro parameter declared with a non-Expr type receives the constant.
class MacroExprDefs {
	static macro function kind(e:Expr) {
		var name = switch e.expr {
			case EBinop(OpAdd, _, _): "add";
			case EBinop(OpAssignOp(OpAdd), _, _): "addassign";
			case EUnop(OpIncrement, true, _): "postinc";
			case EIf(_, _, null): "if-noelse";
			case EIf(_, _, _): "if-else";
			case EWhile(_, _, false): "dowhile";
			case EWhile(_, _, true): "while";
			case EFor({expr: EBinop(OpIn, {expr: EConst(CIdent(v))}, _)}, _): "for " + v;
			case ESwitch(_, cases, _): "switch " + cases.length;
			case EVars([{name: n}]): "var " + n;
			case EFunction(FArrow, f): "arrow " + f.args.length;
			case EFunction(FAnonymous, f): "fn " + f.args.length;
			case EArrayDecl(items): "array " + items.length;
			case EObjectDecl(fields): "object " + fields[0].field;
			case ENew(p, _): "new " + p.name;
			case ECheckType(_, TPath(p)): "check " + p.name;
			case ETernary(_, _, _): "ternary";
			case EParenthesis(_): "paren";
			case EReturn(_): "return";
			case EConst(CString(s, _)): "string " + s;
			case EConst(CFloat(f, _)): "float " + f;
			case _: "other";
		}
		return macro $v{name};
	}

	static macro function build() {
		var sw = {expr: ESwitch(macro 2, [{values: [macro 1], guard: null, expr: macro "one"}, {values: [macro 2], guard: null, expr: macro "two"}], macro "none"), pos: Context.currentPos()};
		return sw;
	}

	static macro function sum() {
		var b = {expr: EBinop(OpAdd, macro 40, {expr: EConst(CInt("2")), pos: Context.currentPos()}), pos: Context.currentPos()};
		return {expr: EParenthesis(b), pos: Context.currentPos()};
	}

	static macro function loop() {
		var body = {expr: EBinop(OpAssignOp(OpAdd), macro acc, macro i), pos: Context.currentPos()};
		var head = {expr: EBinop(OpIn, macro i, macro 0...5), pos: Context.currentPos()};
		var decl = {expr: EVars([{name: "acc", type: null, expr: macro 0}]), pos: Context.currentPos()};
		var forE = {expr: EFor(head, body), pos: Context.currentPos()};
		return {expr: EBlock([decl, forE, macro acc]), pos: Context.currentPos()};
	}

	static macro function tern() {
		return {expr: ETernary(macro true, macro "t", macro "f"), pos: Context.currentPos()};
	}

	static macro function unop() {
		return {expr: EUnop(OpNeg, false, macro 7), pos: Context.currentPos()};
	}

	static macro function constants(a:String, b:Array<Int>, c:{}) {
		return macro $v{a + b[1] + Reflect.field(c, "k")};
	}

	static var failed = false;

	static function check(label:String, got:Dynamic, want:Dynamic) {
		if (got != want) {
			failed = true;
			Sys.println("FAIL " + label + ": got " + got + " want " + want);
		}
	}

	static function main() {
		var x = 1;
		check("add", kind(1 + 2), "add");
		check("addassign", kind(x += 2), "addassign");
		check("postinc", kind(x++), "postinc");
		check("if", kind(if (x > 0) 1), "if-noelse");
		check("ifelse", kind(if (x > 0) 1 else 2), "if-else");
		check("while", kind(while (false) {}), "while");
		check("dowhile", kind(do {} while (false)), "dowhile");
		check("for", kind(for (i in 0...3) {}), "for i");
		check("switch", kind(switch x { case 1: 1; case _: 2; }), "switch 2");
		check("var", kind(var y = 3), "var y");
		check("arrow", kind((a, b) -> a), "arrow 2");
		check("fn", kind(function(a) return a), "fn 1");
		check("array", kind([1, 2, 3]), "array 3");
		check("object", kind({a: 1}), "object a");
		check("new", kind(new StringBuf()), "new StringBuf");
		check("checktype", kind((x : Int)), "check Int");
		check("ternary", kind(x > 0 ? 1 : 2), "ternary");
		check("paren", kind((x)), "paren");
		check("string", kind("hi"), "string hi");
		check("float", kind(1.5), "float 1.5");
		check("build switch", build(), "two");
		check("sum", sum(), 42);
		check("loop", loop(), 10);
		check("ternary built", tern(), "t");
		check("unop built", unop(), -7);
		check("constant params", constants('s$x', [1, 2], {k: "v"}), "s$x2v");
		if (!failed)
			Sys.println("CONFORMANCE_OK");
	}
}
