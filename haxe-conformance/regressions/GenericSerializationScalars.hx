class GenericSerializationScalars {
    public function new() {}
    function copy<T>(value:T):T return haxe.Unserializer.run(haxe.Serializer.run(value));
    static function main() {
        var copier = new GenericSerializationScalars();
        if (!Math.isNaN(copier.copy(Math.NaN))) throw "generic NaN";
        if (copier.copy(Math.POSITIVE_INFINITY) != Math.POSITIVE_INFINITY) throw "generic infinity";
        if (copier.copy(Math.NEGATIVE_INFINITY) != Math.NEGATIVE_INFINITY) throw "generic negative infinity";
        if (copier.copy(1.5) != 1.5 || copier.copy(-2.75) != -2.75) throw "generic float";
        if (copier.copy(-7) != -7) throw "generic int";
        if (copier.copy(true) != true || copier.copy(false) != false) throw "generic boolean";
        if (copier.copy("hello") != "hello") throw "generic string";
        var value:Dynamic = 1.5;
        var copied:Dynamic = copier.copy(value);
        if (Type.typeof(copied) != Type.ValueType.TFloat || copied != value) throw "generic Dynamic";
        Sys.println("CONFORMANCE_OK");
    }
}
