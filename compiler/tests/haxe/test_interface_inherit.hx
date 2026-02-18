interface Drawable {
    public function draw():String;
}

interface Clickable {
    public function click():String;
}

// Child interface extends parent
interface Widget extends Drawable {
    public function resize():String;
}

// Multiple parent interfaces
interface InteractiveWidget extends Widget, Clickable {
    public function focus():String;
}

class Button implements InteractiveWidget {
    public function new() {}

    public function draw():String {
        return "Button.draw";
    }

    public function resize():String {
        return "Button.resize";
    }

    public function click():String {
        return "Button.click";
    }

    public function focus():String {
        return "Button.focus";
    }
}

class Main {
    static function main() {
        var btn = new Button();

        // Direct class usage
        trace(btn.draw());    // Button.draw
        trace(btn.resize());  // Button.resize
        trace(btn.click());   // Button.click
        trace(btn.focus());   // Button.focus

        // Upcast to Widget (single parent)
        var w:Widget = btn;
        trace(w.draw());      // Button.draw
        trace(w.resize());    // Button.resize

        // Upcast to Drawable (grandparent interface)
        var d:Drawable = btn;
        trace(d.draw());      // Button.draw

        // Upcast to Clickable (parent of InteractiveWidget)
        var c:Clickable = btn;
        trace(c.click());     // Button.click

        // Upcast to InteractiveWidget (has all methods)
        var iw:InteractiveWidget = btn;
        trace(iw.draw());     // Button.draw
        trace(iw.resize());   // Button.resize
        trace(iw.click());    // Button.click
        trace(iw.focus());    // Button.focus

        trace("done");
    }
}
