class DateConstructor {
    static function main() {
        var d = new Date(2020, 5, 14, 8, 9, 10);
        if (d.getFullYear() != 2020 || d.getMonth() != 5 || d.getDate() != 14 || d.getHours() != 8 || d.getMinutes() != 9 || d.getSeconds() != 10) throw "bad Date components";
        var copy = Date.fromTime(d.getTime());
        if (copy.getTime() != d.getTime()) throw "bad Date copy";
        Sys.println("CONFORMANCE_OK");
    }
}
