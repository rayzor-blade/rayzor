class StandardStreams {
    static function main() {
        var input:haxe.io.Input = Sys.stdin();
        if (input.readString(0) != "") throw "empty stdin read";
        input.close();
        input.close();
        var output:haxe.io.Output = Sys.stdout();
        output.writeString("native stdout λ\n");
        output.flush();
        output.close();
        output.close();
        var error:haxe.io.Output = Sys.stderr();
        error.writeString("native stderr λ\n");
        error.flush();
        error.close();
        error.close();
        trace("CONFORMANCE_OK");
    }
}
