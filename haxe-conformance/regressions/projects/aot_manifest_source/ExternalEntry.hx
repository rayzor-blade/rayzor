class ExternalEntry {
    static function main() {
        if (ManifestDependency.value() != "manifest dependency") throw "class path";
        trace("CONFORMANCE_OK");
    }
}
