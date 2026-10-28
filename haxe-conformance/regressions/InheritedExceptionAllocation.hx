import haxe.exceptions.NotImplementedException;

class InheritedExceptionAllocation {
    static function main() {
        for (i in 0...64) {
            try {
                throw new NotImplementedException("message " + i);
            } catch (e:NotImplementedException) {
                if (e.message != "message " + i) throw "message";
                if (e.posInfos.className != "InheritedExceptionAllocation") throw "position";
                if (e.stack == null) throw "stack";
                var values = [i, i + 1, i + 2];
                if (values[2] != i + 2) throw "array";
            }
        }
        Sys.println("CONFORMANCE_OK");
    }
}
