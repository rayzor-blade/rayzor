private class AssignmentBox {
    public var stored:Int = 4;
    public var reads:Int = 0;
    public var writes:Int = 0;
    public var value(get, set):Int;
    public var readOnly(get, default):Int = 7;
    public var backed(default, set):Int = 2;
    function set_backed(v:Int):Int { throw "bypassed setter ran"; }
    public function new() {}
    function get_value():Int { reads++; return stored; }
    function set_value(v:Int):Int { writes++; stored = v; return v + 100; }
    function get_readOnly():Int { reads++; return readOnly; }
}
private abstract ScalarProperty(Int) from Int {
    public var value(get, set):Int;
    inline function get_value():Int { return this; }
    inline function set_value(v:Int):Int { return this = v; }
}
private abstract RecordProperty({stored:Int, reads:Int, writes:Int}) {
    public var value(get, set):Int;
    public function new() { this = {stored: 3, reads: 0, writes: 0}; }
    function get_value():Int { this.reads++; return this.stored; }
    function set_value(v:Int):Int { this.writes++; return this.stored = v; }
    public function check():Bool { return this.stored == 7 && this.reads == 1 && this.writes == 1; }
}
private abstract Accumulator({value:Int}) {
    public function new() this = {value: 2};
    @:op(A += B) public inline function add(n:Int):Int return this.value += n;
    public function read():Int return this.value;
}
class AssignmentEvaluation {
    static var calls:Int = 0;
    static var boxes:Array<AssignmentBox>;
    static var accumulators:Array<Accumulator>;
    static function getAccumulators():Array<Accumulator> { calls++; return accumulators; }
    static var staticStored:Int = 3;
    static var staticValue(get, set):Int;
    static function get_staticValue():Int { return staticStored; }
    static function set_staticValue(v):Int { staticStored = v; return v + 10; }
    static function getBoxes():Array<AssignmentBox> { calls++; return boxes; }
    static function main() {
        var box = new AssignmentBox();
        boxes = [box];
        var index = 0;
        var result = getBoxes()[index++].value += 3;
        if (result != 107 || box.stored != 7) throw "compound result";
        if (calls != 1 || index != 1 || box.reads != 1 || box.writes != 1) throw "compound evaluation";
        index = 0;
        result = getBoxes()[index++].value = 9;
        if (result != 109 || box.stored != 9) throw "assignment result";
        if (calls != 2 || index != 1 || box.reads != 1 || box.writes != 2) throw "assignment evaluation";
        index = 0;
        result = getBoxes()[index++].readOnly += 2;
        if (result != 9 || box.readOnly != 9) throw "getter with backing write";
        if (calls != 3 || index != 1 || box.reads != 3) throw "getter count";
        index = 0;
        result = ++getBoxes()[index++].value;
        if (result != 110 || box.stored != 10 || index != 1 || calls != 4) throw "prefix result";
        index = 0;
        result = getBoxes()[index++].value++;
        if (result != 10 || box.stored != 11 || index != 1 || calls != 5) throw "postfix result";
        if (box.reads != 5 || box.writes != 4) throw "increment accessors";
        if ((staticValue += 2) != 15 || staticStored != 5) throw "static compound result";
        if (++staticValue != 16 || staticStored != 6) throw "static prefix result";
        if (staticValue++ != 6 || staticStored != 7) throw "static postfix result";
        if ((staticValue = index = 8) != 18 || index != 8 || staticStored != 8) throw "chained setter assignment";
        @:bypassAccessor box.backed = 12;
        if (box.backed != 12) throw "bypass write";
        var readsBefore = box.reads;
        var raw = @:bypassAccessor box.readOnly;
        if (raw != 9 || box.reads != readsBefore) throw "bypass read";
        var object = { value: 2 };
        if ((object.value += 3) != 5 || object.value != 5) throw "structural identity";
        var numbers = [31];
        index = 0;
        if ((numbers[index++] &= 14) != 14 || index != 1) throw "and assignment";
        index = 0;
        if ((numbers[index++] |= 1) != 15 || index != 1) throw "or assignment";
        index = 0;
        if ((numbers[index++] ^= 3) != 12 || index != 1) throw "xor assignment";
        index = 0;
        if ((numbers[index++] <<= 1) != 24 || index != 1) throw "shift left";
        index = 0;
        if ((numbers[index++] >>= 2) != 6 || index != 1) throw "shift right";
        index = 0;
        if ((numbers[index++] >>>= 1) != 3 || index != 1) throw "unsigned shift";
        var scalar:ScalarProperty = 2;
        if ((scalar.value += 3) != 5 || scalar.value != 5) throw "scalar abstract writeback";
        var scalars:Array<ScalarProperty> = [scalar];
        index = 0;
        if ((scalars[index++].value += 2) != 7 || index != 1 || scalars[0].value != 7) throw "array abstract writeback";
        if (scalar.value != 5) throw "scalar abstract value semantics";
        var record = new RecordProperty();
        if ((record.value += 4) != 7 || !record.check()) throw "record abstract accessors";
        var sum = 0;
        for (v in scalars) sum += v.value;
        if (sum != 7) throw "loop-carried compound assignment";
        accumulators = [new Accumulator()];
        var previousCalls = calls;
        index = 0;
        if ((getAccumulators()[index++] += 3) != 5 || accumulators[0].read() != 5) throw "compound overload result";
        if (index != 1 || calls != previousCalls + 1) throw "compound overload evaluation";
        trace("CONFORMANCE_OK");
    }
}
