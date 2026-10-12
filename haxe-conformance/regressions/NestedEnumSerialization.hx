enum SerializationNestedValue {
    Empty;
    Payload(number:Int, text:String);
    Nested(value:SerializationNestedValue);
    Pair(left:SerializationNestedValue, right:SerializationNestedValue);
}

class NestedEnumSerialization {
    static function main() {
        var value = SerializationNestedValue.Nested(SerializationNestedValue.Nested(SerializationNestedValue.Payload(0, "hello")));
        var restored:SerializationNestedValue = cast haxe.Unserializer.run(haxe.Serializer.run(value));
        if (!Type.enumEq(value, restored)) throw "nested enum round trip";
        var parameters = Type.enumParameters(restored);
        if (Type.getEnum(parameters[0]) != SerializationNestedValue) throw "nested parameter identity";
        var pair = SerializationNestedValue.Pair(SerializationNestedValue.Empty, SerializationNestedValue.Empty);
        var serializer = new haxe.Serializer();
        serializer.useCache = true;
        serializer.serialize(pair);
        var copied:SerializationNestedValue = cast haxe.Unserializer.run(serializer.toString());
        if (!Type.enumEq(pair, copied)) throw "cached enum parameters";
        Sys.println("CONFORMANCE_OK");
    }
}
