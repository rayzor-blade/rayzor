class Main {
    static function main() {
        // Test 1: Non-global map — replace first match only
        var r = ~/([a-z]+)/;
        var result = r.map("123abc456def", function(e:EReg):String {
            return e.matched(0).toUpperCase();
        });
        trace(result); // 123ABC456def

        // Test 2: Global map — replace all matches
        var rg = ~/([a-z]+)/g;
        var result2 = rg.map("123abc456def789ghi", function(e:EReg):String {
            return e.matched(0).toUpperCase();
        });
        trace(result2); // 123ABC456DEF789GHI

        // Test 3: Map with capture groups
        var rp = ~/(\w+)=(\w+)/g;
        var result3 = rp.map("a=1 b=2 c=3", function(e:EReg):String {
            return e.matched(2) + ":" + e.matched(1);
        });
        trace(result3); // 1:a 2:b 3:c

        // Test 4: No match — returns original string
        var rn = ~/xyz/;
        var result4 = rn.map("hello world", function(e:EReg):String {
            return "REPLACED";
        });
        trace(result4); // hello world

        trace("done");
    }
}
