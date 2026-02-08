typedef User = {
    var name:String;
    var age:Int;
};

class Main {
    static function main() {
        var u:User = {name: "Alice", age: 25};
        trace(u.name);   // Alice
        trace(u.age);    // 25
        u.age = 30;
        trace(u.age);    // 30
    }
}
