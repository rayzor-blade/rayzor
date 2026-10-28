import haxe.macro.Expr;

class MacroSystemCommand {
    static macro function commandResult():Expr {
        var code = Sys.systemName() == "Windows"
            ? Sys.command("cmd", ["/C", "exit 7"])
            : Sys.command("/bin/sh", ["-c", "printf macro-host-output; exit 7"]);
        if (code != 7) throw "wrong command exit status";
        if (haxe.io.Path.normalize("a/../b") != "b") throw "wrong stdlib body";
        var bytes = haxe.io.Bytes.ofString("snapshot");
        if (bytes.length != 8 || bytes.toString() != "snapshot") throw "wrong stdlib alias";
        return macro $v{code};
    }

    static function main() {
        if (commandResult() != 7) throw "wrong expansion result";
        Sys.println("CONFORMANCE_OK");
    }
}
