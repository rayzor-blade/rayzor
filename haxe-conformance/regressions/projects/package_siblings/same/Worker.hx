package same;

class Worker {
    public static function code():Int {
        var e = Error.Good;
        return switch (e) {
            case Good: 31;
            case Bad: -1;
        };
    }
}
