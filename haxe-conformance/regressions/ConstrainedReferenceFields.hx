typedef ConstrainedText = String;

class ConstrainedReferenceFields {
    static function stringLength<T:String>(value:T):Int return value.length;
    static function aliasLength<T:ConstrainedText>(value:T):Int return value.length;
    static function arrayLength<T:Array<Int>>(value:T):Int return value.length;

    static function main() {
        if (stringLength("hello") != 5) throw "constrained string";
        if (stringLength("") != 0) throw "empty constrained string";
        if (aliasLength("text") != 4) throw "constrained string alias";
        if (arrayLength([1, 2, 3]) != 3) throw "constrained array";
        if (arrayLength([]) != 0) throw "empty constrained array";
        Sys.println("CONFORMANCE_OK");
    }
}
