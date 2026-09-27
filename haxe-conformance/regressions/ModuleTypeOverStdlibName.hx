package fixtures;

// A type declared in this module wins over a stdlib type of the same simple
// name loaded from elsewhere: Sys brings in haxe.io.Error.
class ModuleTypeOverStdlibName {
    static function main() {
        Sys.print("");
        var unit = "fail";
        try throw new Error("success") catch (e:Error) unit = e.message;
        if (unit != "success") throw "own Error class: " + unit;
        var e = new Error("m");
        var s:String = e.message;
        if (s != "m") throw "message";
        trace("CONFORMANCE_OK");
    }
}

private class Error extends haxe.Exception {}
