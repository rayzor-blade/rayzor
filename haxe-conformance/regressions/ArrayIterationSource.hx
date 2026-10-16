class ArrayIterationSource {
    static var calls = 0;
    static var values = [1, 2, 3];
    static var source(get, never):Array<Int>;
    static function get_source():Array<Int> {
        calls++;
        return values;
    }
    static function make():Array<Int> {
        calls++;
        return [1, 2, 3];
    }
    var data = [1, 2, 3];
    var instance(get, never):Array<Int>;
    function get_instance():Array<Int> {
        calls++;
        return [1, 2, 3];
    }
    public function new() {}
    function iterateInstance():Int {
        var sum = 0;
        for (value in instance) sum += value;
        return sum;
    }
    function iterateField():Int {
        var sum = 0;
        for (value in data) {
            data = [9, 9, 9];
            sum += value;
        }
        return sum;
    }
    static function main():Void {
        var sum = 0;
        for (value in make()) {
            if (value == 2) continue;
            sum += value;
        }
        if (calls != 1 || sum != 4) throw "call source";
        sum = 0;
        for (value in source) {
            values = [9, 9, 9];
            sum += value;
        }
        if (calls != 2 || sum != 6) throw "getter source";
        var local = [1, 2, 3];
        sum = 0;
        for (value in local) {
            local = [9, 9, 9];
            sum += value;
        }
        if (sum != 19) throw "local source reassignment";
        sum = 0;
        for (outer in make()) {
            for (inner in make()) sum += outer * inner;
        }
        if (calls != 6 || sum != 36) throw "nested sources";
        sum = 0;
        for (index => value in make()) sum += index + value;
        if (calls != 7 || sum != 9) throw "key value source";
        if (new ArrayIterationSource().iterateInstance() != 6 || calls != 8) throw "instance getter source";
        values = [1, 2, 3];
        sum = 0;
        for (value in values) {
            values = [9, 9, 9];
            sum += value;
        }
        if (sum != 6 || new ArrayIterationSource().iterateField() != 6) throw "field source identity";
        trace("CONFORMANCE_OK");
    }
}
