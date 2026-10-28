class Main {
    static function main() {
        for (i in 0...64) {
            var value = Root.make(i);
            if (value.count != i || value.text != "root" || value.doubled != i * 2
                || value.branch != i + 1 || value.leaf != i + 2)
                throw "inherited layout";
        }
        Sys.println("CONFORMANCE_OK");
    }
}
