@:rtti private class Described {
    public static var count:Int = 0xAB;
    public var word:String = 'hello';
    public function new() {}
    public static function defaults(text='word', number=0xCD, ratio=1.25, enabled=true):Void {}
}
class RttiDefaults {
    static function main() {
        if (!haxe.rtti.Rtti.hasRtti(Described)) throw "missing RTTI";
        var info = haxe.rtti.Rtti.getRtti(Described);
        var count = false;
        var method = false;
        for (field in info.statics) {
            if (field.name == 'count') {
                if (field.expr != '0xAB') throw 'integer spelling';
                count = true;
            }
            if (field.name == 'defaults') switch field.type {
                case CFunction(args, _):
                    if (args.length != 4 || args[0].value != '"word"' || args[1].value != '0xCD' || args[2].value != '1.25' || args[3].value != 'true') throw 'argument defaults';
                    if (!args[0].opt || args[0].name != 'text') throw 'argument description';
                    method = true;
                case _: throw 'function type';
            }
        }
        if (!count || !method) throw 'missing members';
        trace('CONFORMANCE_OK');
    }
}
