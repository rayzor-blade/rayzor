class Main {
    static function main() {
        // Test map
        var arr = [1, 2, 3, 4, 5];
        var doubled = arr.map(function(x) return x * 2);
        trace(doubled[0]); // 2
        trace(doubled[1]); // 4
        trace(doubled[4]); // 10

        // Test filter
        var evens = arr.filter(function(x) return x % 2 == 0);
        trace(evens.length); // 2
        trace(evens[0]); // 2
        trace(evens[1]); // 4

        // Test sort (ascending)
        var unsorted = [5, 3, 1, 4, 2];
        unsorted.sort(function(a, b) return a - b);
        trace(unsorted[0]); // 1
        trace(unsorted[1]); // 2
        trace(unsorted[4]); // 5
    }
}
