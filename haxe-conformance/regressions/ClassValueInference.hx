import haxe.macro.Context;
import haxe.macro.Expr;

class ConstructedRecord {
    public var value:Int;
    public function new(value:Int) this.value = value;
    public function get():Int return value;
}

class ClassValueInference {
    public static macro function typeName(expression:Expr):Expr {
        var name = haxe.macro.TypeTools.toString(Context.typeof(expression));
        return macro $v{name};
    }

    #if !macro
    static function construct<T>(type:Class<T>, value:Int):T {
        return Type.createInstance(type, [value]);
    }

    static function main() {
        if (typeName(Type.createInstance(ConstructedRecord, [17])) != "ConstructedRecord") {
            throw "constructor return type";
        }
        if (typeName(construct(ConstructedRecord, 23)) != "ConstructedRecord") {
            throw "generic class parameter";
        }
        if (typeName(Type.createEmptyInstance(ConstructedRecord)) != "ConstructedRecord") {
            throw "empty constructor return type";
        }
        var record = Type.createInstance(ConstructedRecord, [17]);
        if (record.get() != 17) throw "constructed method";
        if (Type.getClass(record) != ConstructedRecord) throw "constructed identity";
        var other = construct(ConstructedRecord, 23);
        if (other.get() != 23) throw "generic construction";
        var type:Class<ConstructedRecord> = ConstructedRecord;
        var stored = Type.createInstance(type, [31]);
        if (stored.get() != 31) throw "stored class value";
        Sys.println("CONFORMANCE_OK");
    }
    #end
}
