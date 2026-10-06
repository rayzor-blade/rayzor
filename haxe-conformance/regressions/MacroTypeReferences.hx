import haxe.macro.Context;
import haxe.macro.Expr;
using haxe.macro.TypeTools;

@:mark("example") private abstract ReferenceWrapper<T>(Array<T>) {}
private typedef ReferenceAlias<T> = Array<T>;
private typedef ReferenceMaybeName = Null<String>;
private enum abstract ReferenceColor(Int) {
    var Red = 3;
    var Blue = 7;
    public static function count() return 2;
}
private class ReferenceCarrier {
    @:mark("field") public var name:String;
    public static var count:Int = 1;
    public function new() {}
}

class MacroTypeReferences {
    static macro function describe(e:Expr) {
        var t = Context.typeof(e);
        var result = switch t {
            case TAbstract(ref, args):
                var a = ref.get();
                a.name + ":" + a.meta.has(":notNull") + ":" + a.meta.has(":mark");
            case TType(ref, args):
                ref.get().name + ":" + t.follow().toString();
            case TInst(ref, args): ref.get().name;
            default: "other";
        }
        return macro $v{result};
    }
    static macro function followed(e:Expr) {
        return macro $v{Context.typeof(e).followWithAbstracts().toString()};
    }
    static macro function substituted(e:Expr) {
        var result = switch Context.typeof(e) {
            case TType(ref, args):
                var td = ref.get();
                TypeTools.applyTypeParameters(td.type, td.params, args).toString();
            case TAbstract(ref, args):
                var ab = ref.get();
                TypeTools.applyTypeParameters(ab.type, ab.params, args).toString();
            default: "other";
        }
        return macro $v{result};
    }
    static macro function carrierFields() {
        var result = [];
        switch Context.getType("ReferenceCarrier") {
            case TInst(ref, _):
                var c = ref.get();
                for (f in c.fields.get()) {
                    if (f.name == "name") result.push(f.name + ":" + f.type.toString() + ":" + f.meta.has(":mark"));
                }
                for (f in c.statics.get()) {
                    if (f.name == "count") result.push(f.name + ":" + f.type.toString());
                }
            default:
        }
        return macro $v{result.join(",")};
    }
    static macro function mapped(e:Expr) {
        var result = Context.typeof(e).map(function(_) return Context.getType("String"));
        var text = switch result {
            case TAnonymous(ref):
                var fields = ref.get().fields;
                for (f in fields) if (f.type.toString() != "String") throw "unmapped field";
                "fields:" + fields.length;
            default: result.toString();
        }
        return macro $v{text};
    }
    static macro function quotedType(e:Expr) {
        var t = Context.typeof(e);
        return macro {label:$v{t.follow().toString()}};
    }
    static macro function colorValues() {
        var values = [];
        switch Context.getType("ReferenceColor").follow() {
            case TAbstract(_.get() => ab, _) if (ab.meta.has(":enum")):
                for (field in ab.impl.get().statics.get()) {
                    if (field.meta.has(":enum") && field.meta.has(":impl")) {
                        var name = field.name;
                        values.push(macro ReferenceColor.$name);
                    }
                }
            default: throw "missing enum abstract";
        }
        return macro $a{values};
    }
    static function check(actual:String, expected:String) {
        if (actual != expected) throw actual + " != " + expected;
    }
    static function main() {
        check(describe(1), "Int:true:false");
        check(describe((null:Null<Int>)), "Null:false:false");
        check(describe((null:ReferenceWrapper<Int>)), "ReferenceWrapper:false:true");
        check(describe((null:ReferenceMaybeName)), "ReferenceMaybeName:String");
        check(describe((null:ReferenceCarrier)), "ReferenceCarrier");
        check(describe("hello"), "String");
        check(describe([1]), "Array");
        check(describe((null:ReferenceAlias<String>)), "ReferenceAlias:Array<String>");
        check(followed((null:ReferenceWrapper<String>)), "Array<String>");
        check(substituted((null:ReferenceWrapper<String>)), "Array<String>");
        check(substituted((null:ReferenceAlias<Int>)), "Array<Int>");
        check(carrierFields(), "name:String:true,count:Int");
        check(mapped([1]), "Array<String>");
        check(mapped({foo:1, bar:true}), "fields:2");
        check(quotedType([1]).label, "Array<Int>");
        var colors = colorValues();
        if (colors.length != 2 || colors[0] != ReferenceColor.Red || colors[1] != ReferenceColor.Blue) throw "bad enum values";
        Sys.println("CONFORMANCE_OK");
    }
}
