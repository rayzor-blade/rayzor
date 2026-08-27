@:structInit
private class StructInitPerson {
    public var fullName:String;
    public var firstName(get, never):String;

    function get_firstName():String {
        return fullName.split(" ")[0];
    }
}

class TestStructInitObjectLayout {
    static function main() {
        var person:StructInitPerson = {fullName: "John Smith"};
        if (person.firstName != "John") {
            trace("FAIL: struct-init class used an incompatible object layout");
            return;
        }
        trace("PASS");
    }
}
