class SysCommandArguments {
    static function main():Void {
        var python = Sys.systemName() == "Windows" ? "python" : "python3";
        var literal = "a b; $(exit 99) & \"quoted\" \\ unicode 😂";
        var args = ["-c", "import sys; assert sys.argv[1:] == [r'a b; $(exit 99) & \"quoted\" \\ unicode 😂', '']; sys.exit(42)", literal, ""];
        if (Sys.command(python, args) != 42) throw "literal arguments or exit status";

        var erased:Array<Dynamic> = ["-c", "import sys; assert sys.argv[1:] == ['boxed argument']; sys.exit(44)", "boxed argument"];
        var strings:Array<String> = cast erased;
        if (Sys.command(python, strings) != 44) throw "boxed String arguments";

        var exit = Sys.systemName() == "Windows" ? "exit /b " : "exit ";
        if (Sys.command(exit + "41") != 41) throw "omitted shell arguments";
        if (Sys.command(exit + "43", null) != 43) throw "null shell arguments";
        if (Sys.systemName() != "Windows" && Sys.command("true", []) != 0) throw "empty arguments";
        trace("CONFORMANCE_OK");
    }
}
