class Main {
    static function main() {
        var arch = Sys.cpuArch();
        var ok = arch == "aarch64" || arch == "x86_64" || arch == "x86" || arch == "wasm32";
        if (!ok) {
            throw "unexpected cpu arch: " + arch;
        }
        Sys.println("PASS sys-cpu-arch arch=" + arch);
    }
}
