import sys.io.Process;
class ProcessPipes {
    static function main() {
        var python = Sys.systemName() == "Windows" ? "python" : "python3";
        var process = new Process(python, ["-c", "import sys; data = sys.stdin.buffer.read(); sys.stdout.buffer.write(data); sys.stderr.write('stderr'); sys.exit(7)"]);
        if (process.getPid() <= 0) throw "pid";
        if (process.exitCode(false) != null) throw "early exit";
        var value = "quoted ' \" $HOME; echo wrong — 中文\n";
        process.stdin.writeString(value);
        process.stdin.flush();
        process.stdin.close();
        process.stdin.close();
        if (process.stdout.readAll().toString() != value) throw "stdout";
        if (process.stderr.readAll().toString() != "stderr") throw "stderr";
        if (process.exitCode() != 7 || process.exitCode(false) != 7) throw "exit";
        process.stdout.close();
        process.close();
        process.close();

        var literal = new Process(python, ["-c", "import sys; sys.stdout.write(sys.argv[1])", value]);
        if (literal.stdout.readAll().toString() != value) throw "arguments";
        if (literal.exitCode() != 0) throw "argument exit";
        literal.close();

        var shell = new Process("echo process-shell");
        if (StringTools.trim(shell.stdout.readAll().toString()) != "process-shell") throw "shell";
        if (shell.exitCode() != 0) throw "shell exit";
        shell.close();
        trace("CONFORMANCE_OK");
    }
}
