// A loop's values used after it, and reads of a shorter array guarded by its
// length, must come out the same however the loop is copied.
class Main {
    static function subtract(a:Array<Int>, b:Array<Int>):Int {
        var borrow = 0;
        for (i in 0...a.length) {
            var v = a[i] - (i < b.length ? b[i] : 0) - borrow;
            if (v < 0) {
                v += 10;
                borrow = 1;
            } else {
                borrow = 0;
            }
            a[i] = v;
        }
        return borrow;
    }

    static function scale(a:Array<Int>, m:Int):Int {
        var carry = 0;
        for (i in 0...a.length) {
            var v = a[i] * m + carry;
            a[i] = v % 10;
            carry = Std.int(v / 10);
        }
        var extra = 0;
        while (carry > 0) {
            extra++;
            carry = Std.int(carry / 10);
        }
        return extra;
    }

    static function nested(rows:Array<Array<Int>>):Int {
        var total = 0;
        var last = -1;
        for (r in 0...rows.length) {
            var row = rows[r];
            var sum = 0;
            for (j in 0...row.length)
                sum += row[j];
            last = sum;
            total += sum;
        }
        return total * 1000 + last;
    }

    static function main() {
        var a = [3, 0, 0, 1];
        if (subtract(a, [5, 1]) != 0 || a.join(",") != "8,8,9,0") throw "subtract " + a.join(",");
        var b = [1, 2];
        if (subtract(b, [4, 5, 6]) != 1 || b.join(",") != "7,6") throw "subtract longer " + b.join(",");
        var c = [9, 9, 9];
        if (scale(c, 7) != 1 || c.join(",") != "3,9,9") throw "scale " + c.join(",");
        if (scale([], 7) != 0) throw "scale empty";
        if (nested([[1, 2], [3], [], [4, 5, 6]]) != 21015) throw "nested " + nested([[1, 2], [3], [], [4, 5, 6]]);
        if (nested([]) != -1) throw "nested empty";
        Sys.println("CONFORMANCE_OK");
    }
}
