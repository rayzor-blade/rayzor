import haxe.ds.IntMap;

class Main {
    static function main() {
        // Test 1: key => value iteration on IntMap (literal syntax)
        var ages = [1 => 10, 2 => 20, 3 => 30];

        var sum = 0;
        for (key => value in ages) {
            sum += value;
        }
        trace(sum);  // 60

        // Test 2: key => value iteration on StringMap (literal syntax)
        var scores = ["math" => 95, "english" => 87];

        var total = 0;
        for (subject => score in scores) {
            total += score;
        }
        trace(total);  // 182

        // Test 3: key => value iteration on IntMap via constructor
        var items = new IntMap();
        items.set(10, 100);
        items.set(20, 200);

        var itemSum = 0;
        for (k => v in items) {
            itemSum += v;
        }
        trace(itemSum);  // 300

        trace("done");
    }
}
