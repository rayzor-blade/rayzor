class Main {
    static function main() {
        #if rayzor
        trace("rayzor");
        #else
        trace("other");
        #end
        trace("done");
    }
}
