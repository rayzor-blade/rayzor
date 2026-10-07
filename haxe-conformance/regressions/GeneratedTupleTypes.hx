import haxe.macro.Context;
import haxe.macro.Expr;
import haxe.macro.Type;
using haxe.macro.Tools;

#if !macro
@:genericBuild(GeneratedTupleTypes.TupleBuild.buildTuple())
class GeneratedTuple<Rest> {}
#end

class GeneratedTupleTypes {
    macro static function sameType(value:Expr, expected:Expr):Expr {
        var actual = Context.typeof(value).toString();
        var target = Context.typeof(expected).toString();
        if (actual != target) Context.error(actual + " != " + target, value.pos);
        return macro true;
    }
    #if !macro
    static function main() {
        var explicit = new GeneratedTuple<String, Int>("foo", 12);
        if (!sameType(explicit, (null:GeneratedTuple<String, Int>))) throw "explicit type";
        if (explicit.v0 != "foo" || explicit.v1 != 12) throw "explicit fields";

        var inferred = new GeneratedTuple("bar", 34);
        if (!sameType(inferred, (null:GeneratedTuple<String, Int>))) throw "inferred type";
        if (inferred.v0 != "bar" || inferred.v1 != 34) throw "inferred fields";

        var annotated:GeneratedTuple = new GeneratedTuple("baz", 56);
        if (!sameType(annotated, (null:GeneratedTuple<String, Int>))) throw "annotated type";
        if (annotated.v0 != "baz" || annotated.v1 != 56) throw "annotated fields";

        final frozen:GeneratedTuple = new GeneratedTuple(2.5, true);
        if (!sameType(frozen, (null:GeneratedTuple<Float, Bool>))) throw "final type";
        if (frozen.v0 != 2.5 || !frozen.v1) throw "final fields";

        var single = new GeneratedTuple<Array<Int>>([8, 9]);
        if (single.v0[1] != 9) throw "array field";
        var triple = new GeneratedTuple("three", 3, false);
        if (triple.v0 != "three" || triple.v1 != 3 || triple.v2) throw "three fields";
        if (explicit.v1 != 12) throw "cached type parameters";
        Sys.println("CONFORMANCE_OK");
    }
    #end
}

class TupleBuild {
	#if macro
	static var tupleMap = new Map();
	#end

	macro static public function buildTuple():ComplexType {
		switch (Context.getLocalType()) {
			case TInst(c, args):
				var arity = args.length;
				if (arity == 0) {
					var el = Context.getCallArguments();
					if (el != null && el.length > 0) {
						args = [for (e in el) Context.typeof(e)];
						arity = args.length;
					} else {
						return null;
					}
				}
				if (!tupleMap.exists(arity)) {
					tupleMap[arity] = buildTupleType(c.get(), Context.getBuildFields(), arity);
				}
				var ct = tupleMap[arity];
				ct.params = [
					for (t in args) {
						switch (t) {
							case TInst(_.get().kind => KExpr(e), _):
								TPType(Context.typeof(e).toComplexType());
							case _:
								TPType(t.toComplexType());
						}
					}
				];
				return TPath(ct);
			case _:
				return Context.error("Class expected", Context.currentPos());
		}
	}

	#if macro
	static function buildTupleType(c:ClassType, fields:Array<Field>, arity:Int):TypePath {
		var typeParams = [];
		var tupleFields = [];
		for (i in 0...arity) {
			var fieldName = 'v$i';
			var typeParamName = 'T$i';
			var typeParam = {
				TPath({
					pack: [],
					name: typeParamName,
					sub: null,
					params: []
				});
			}
			typeParams.push({
				name: typeParamName,
				constraints: [],
				params: []
			});
			var field = (macro class X {
				public var $fieldName:$typeParam;
			}).fields[0];
			tupleFields.push(field);
		}
		var constructor = {
			name: "new",
			pos: c.pos,
			access: [APublic, AInline],
			kind: FFun({
				ret: null,
				expr: macro $b{
					tupleFields.map(function(field) {
						var name = field.name;
						return macro this.$name = $i{name};
					})
				},
				params: [],
				args: tupleFields.map(function(field) {
					return {
						name: field.name,
						type: null,
						opt: false,
						value: null,
						meta: []
					}
				})
			})
		}
		var name = c.name + "_" + arity;
		var tDef = {
			pack: c.pack,
			name: name,
			pos: c.pos,
			kind: TDClass(),
			params: typeParams,
			fields: fields.concat(tupleFields).concat([constructor])
		}
		Context.defineType(tDef);
		return {
			pack: c.pack,
			name: name,
			params: [],
			sub: null
		}
	}
	#end
}
