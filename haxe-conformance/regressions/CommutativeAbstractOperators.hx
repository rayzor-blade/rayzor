private abstract NumberWord(Int) from Int {
    @:op(A == B)
    @:commutative
    public function equalsWord(word:String):Bool {
        return Std.string(this) == word;
    }
}

class CommutativeAbstractOperators {
    static function main() {
        var value:NumberWord = 12;
        if (!(value == "12")) throw "abstract on left";
        if (!("12" == value)) throw "abstract on right";
        if ("13" == value) throw "unequal operands";
        var order = "";
        function word():String {
            order += "L";
            return "12";
        }
        function number():NumberWord {
            order += "R";
            return value;
        }
        if (!(word() == number())) throw "commutative result";
        if (order != "LR") throw "operand evaluation order";
        Sys.println("CONFORMANCE_OK");
    }
}
