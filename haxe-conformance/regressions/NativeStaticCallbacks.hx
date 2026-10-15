class NativeStaticCallbacks {
    static function invoke(f:String->Bool, path:String):Bool return f(path);
    static function stringify(f:Dynamic->String, value:Dynamic):String return f(value);
    static function main() {
        var cwd = Sys.getCwd();
        if (!StringTools.endsWith(cwd, "/") && !StringTools.endsWith(cwd, "\\")) throw "cwd separator";
        if (!invoke(sys.FileSystem.exists, cwd)) throw "exists callback";
        if (!invoke(sys.FileSystem.isDirectory, cwd)) throw "directory callback";
        if (invoke(sys.FileSystem.exists, cwd + "missing-native-callback-file")) throw "missing callback";
        if (stringify(Std.string, "text") != "text") throw "Dynamic string callback";
        if (stringify(Std.string, 42) != "42") throw "Dynamic integer callback";
        trace("CONFORMANCE_OK");
    }
}
