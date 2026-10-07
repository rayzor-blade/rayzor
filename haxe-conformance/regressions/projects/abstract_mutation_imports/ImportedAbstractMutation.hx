import helpers.NumericCursor;

class ImportedAbstractMutation {
    static function main():Void {
        var cursor:NumericCursor = 4;
        if (cursor.advance(2) != 4 || cursor != 6) throw "imported writeback";
        var sum = 0;
        for (value in (0:NumericCursor)) sum += value;
        if (sum != 3) throw "imported abstract iteration";
        var values = [for (value in (0:NumericCursor)) value * 2];
        if (values.length != 3 || values[0] != 0 || values[1] != 2 || values[2] != 4) throw "imported comprehension";
        Sys.println("CONFORMANCE_OK");
    }
}
