package app;

import lib.pk.*;

class User {
    public static var tokens:Tokens = { gap: Spacing.defaults() * 5 };

    public static function run():Int {
        return Spacing.defaults() + Other.value() + tokens.gap;
    }
}
