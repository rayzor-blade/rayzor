interface IDrawable {
    public function draw():String;
}

interface IClickable {
    public function click():String;
}

class Widget implements IDrawable {
    public var name:String;
    public function new(n:String) {
        name = n;
    }
    public function draw():String {
        return "draw:" + name;
    }
}

class Button extends Widget implements IClickable {
    public function new(n:String) {
        super(n);
    }
    override public function draw():String {
        return "button-draw:" + name;
    }
    public function click():String {
        return "click:" + name;
    }
}

class Main {
    static function main() {
        // Test 1: Class → Interface (success - Widget implements IDrawable)
        var w:Widget = new Widget("box");
        var d:IDrawable = cast(w, IDrawable);
        if (d != null) {
            trace(d.draw());  // draw:box
        } else {
            trace("fail1");
        }

        // Test 2: Class → Interface (fail - Widget does NOT implement IClickable)
        var c:IClickable = cast(w, IClickable);
        if (c != null) {
            trace("fail2");
        } else {
            trace("null");  // null
        }

        // Test 3: Subclass → Interface (success - Button implements IClickable)
        var btn:Button = new Button("submit");
        var c2:IClickable = cast(btn, IClickable);
        if (c2 != null) {
            trace(c2.click());  // click:submit
        } else {
            trace("fail3");
        }

        // Test 4: Subclass → parent Interface (success - Button inherits IDrawable from Widget)
        var d2:IDrawable = cast(btn, IDrawable);
        if (d2 != null) {
            trace(d2.draw());  // button-draw:submit
        } else {
            trace("fail4");
        }

        trace("done");
    }
}
