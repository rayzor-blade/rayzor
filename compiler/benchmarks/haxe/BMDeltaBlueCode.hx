// DeltaBlue Constraint Solver Benchmark
// Ported from Dart (originally Smalltalk -> JavaScript -> Dart by Google 2008-2010)
// Used to run official Haxe targets (--interp, HashLink, C++) for comparison

class BMStrength {
    public static var REQUIRED = new BMStrength(0, "required");
    public static var STRONG_PREFERRED = new BMStrength(1, "strongPreferred");
    public static var PREFERRED = new BMStrength(2, "preferred");
    public static var STRONG_DEFAULT = new BMStrength(3, "strongDefault");
    public static var NORMAL = new BMStrength(4, "normal");
    public static var WEAK_DEFAULT = new BMStrength(5, "weakDefault");
    public static var WEAKEST = new BMStrength(6, "weakest");

    public var value:Int;
    public var name:String;

    public function new(v:Int, n:String) {
        value = v;
        name = n;
    }

    public function nextWeaker():BMStrength {
        if (value == 0) return WEAKEST;
        if (value == 1) return WEAK_DEFAULT;
        if (value == 2) return NORMAL;
        if (value == 3) return STRONG_DEFAULT;
        if (value == 4) return PREFERRED;
        if (value == 5) return STRONG_PREFERRED;
        return WEAKEST;
    }

    public static function stronger(s1:BMStrength, s2:BMStrength):Bool {
        return s1.value < s2.value;
    }

    public static function weaker(s1:BMStrength, s2:BMStrength):Bool {
        return s1.value > s2.value;
    }

    public static function weakest(s1:BMStrength, s2:BMStrength):BMStrength {
        return weaker(s1, s2) ? s1 : s2;
    }

    public static function strongest(s1:BMStrength, s2:BMStrength):BMStrength {
        return stronger(s1, s2) ? s1 : s2;
    }
}

class BMVariable {
    public var constraints:Array<BMConstraint>;
    public var determinedBy:BMConstraint;
    public var mark:Int;
    public var walkStrength:BMStrength;
    public var stay:Bool;
    public var value:Int;
    public var name:String;

    public function new(n:String, v:Int) {
        name = n;
        value = v;
        constraints = new Array<BMConstraint>();
        determinedBy = null;
        mark = 0;
        walkStrength = BMStrength.WEAKEST;
        stay = true;
    }

    public function addConstraint(c:BMConstraint):Void {
        constraints.push(c);
    }

    public function removeConstraint(c:BMConstraint):Void {
        var newConstraints = new Array<BMConstraint>();
        for (i in 0...constraints.length) {
            if (constraints[i] != c) {
                newConstraints.push(constraints[i]);
            }
        }
        constraints = newConstraints;
        if (determinedBy == c) determinedBy = null;
    }
}

class BMConstraint {
    public var strength:BMStrength;

    public function new(s:BMStrength) {
        strength = s;
    }

    public function isSatisfied():Bool { return false; }
    public function markUnsatisfied():Void {}
    public function addToGraph():Void {}
    public function removeFromGraph():Void {}
    public function chooseMethod(mark:Int):Void {}
    public function markInputs(mark:Int):Void {}
    public function inputsKnown(mark:Int):Bool { return false; }
    public function output():BMVariable { return null; }
    public function execute():Void {}
    public function recalculate():Void {}
    public function isInput():Bool { return false; }

    public function addConstraint():Void {
        addToGraph();
        BMDeltaBlueCode.planner.incrementalAdd(this);
    }

    public function satisfy(mark:Int):BMConstraint {
        chooseMethod(mark);
        if (!isSatisfied()) {
            if (strength == BMStrength.REQUIRED) {
                trace("Could not satisfy a required constraint!");
            }
            return null;
        }
        markInputs(mark);
        var out = output();
        var overridden = out.determinedBy;
        if (overridden != null) overridden.markUnsatisfied();
        out.determinedBy = this;
        if (!BMDeltaBlueCode.planner.addPropagate(this, mark)) trace("Cycle encountered");
        out.mark = mark;
        return overridden;
    }

    public function destroyConstraint():Void {
        if (isSatisfied()) BMDeltaBlueCode.planner.incrementalRemove(this);
        removeFromGraph();
    }
}

class BMUnaryConstraint extends BMConstraint {
    public var myOutput:BMVariable;
    public var satisfied:Bool;

    public function new(output:BMVariable, s:BMStrength) {
        super(s);
        myOutput = output;
        satisfied = false;
        addConstraint();
    }

    override public function addToGraph():Void {
        myOutput.addConstraint(this);
        satisfied = false;
    }

    override public function chooseMethod(mark:Int):Void {
        satisfied = (myOutput.mark != mark) && BMStrength.stronger(strength, myOutput.walkStrength);
    }

    override public function isSatisfied():Bool { return satisfied; }
    override public function markInputs(mark:Int):Void {}
    override public function output():BMVariable { return myOutput; }

    override public function recalculate():Void {
        myOutput.walkStrength = strength;
        myOutput.stay = !isInput();
        if (myOutput.stay) execute();
    }

    override public function markUnsatisfied():Void {
        satisfied = false;
    }

    override public function inputsKnown(mark:Int):Bool { return true; }

    override public function removeFromGraph():Void {
        if (myOutput != null) myOutput.removeConstraint(this);
        satisfied = false;
    }
}

class BMStayConstraint extends BMUnaryConstraint {
    public function new(v:BMVariable, str:BMStrength) {
        super(v, str);
    }

    override public function execute():Void {}
}

class BMEditConstraint extends BMUnaryConstraint {
    public function new(v:BMVariable, str:BMStrength) {
        super(v, str);
    }

    override public function isInput():Bool { return true; }
    override public function execute():Void {}
}

class BMBinaryConstraint extends BMConstraint {
    public var v1:BMVariable;
    public var v2:BMVariable;
    public var direction:Int;

    public static inline var NONE = 1;
    public static inline var FORWARD = 2;
    public static inline var BACKWARD = 0;

    public function new(a:BMVariable, b:BMVariable, s:BMStrength) {
        super(s);
        v1 = a;
        v2 = b;
        direction = NONE;
        addConstraint();
    }

    override public function chooseMethod(mark:Int):Void {
        if (v1.mark == mark) {
            direction = (v2.mark != mark && BMStrength.stronger(strength, v2.walkStrength))
                ? FORWARD : NONE;
        }
        if (v2.mark == mark) {
            direction = (v1.mark != mark && BMStrength.stronger(strength, v1.walkStrength))
                ? BACKWARD : NONE;
        }
        if (BMStrength.weaker(v1.walkStrength, v2.walkStrength)) {
            direction = BMStrength.stronger(strength, v1.walkStrength)
                ? BACKWARD : NONE;
        } else {
            direction = BMStrength.stronger(strength, v2.walkStrength)
                ? FORWARD : BACKWARD;
        }
    }

    override public function addToGraph():Void {
        v1.addConstraint(this);
        v2.addConstraint(this);
        direction = NONE;
    }

    override public function isSatisfied():Bool { return direction != NONE; }

    override public function markInputs(mark:Int):Void {
        input().mark = mark;
    }

    public function input():BMVariable {
        return direction == BMBinaryConstraint.FORWARD ? v1 : v2;
    }

    override public function output():BMVariable {
        return direction == BMBinaryConstraint.FORWARD ? v2 : v1;
    }

    override public function recalculate():Void {
        var ihn = input();
        var out = output();
        out.walkStrength = BMStrength.weakest(strength, ihn.walkStrength);
        out.stay = ihn.stay;
        if (out.stay) execute();
    }

    override public function markUnsatisfied():Void {
        direction = NONE;
    }

    override public function inputsKnown(mark:Int):Bool {
        var i = input();
        return i.mark == mark || i.stay || i.determinedBy == null;
    }

    override public function removeFromGraph():Void {
        if (v1 != null) v1.removeConstraint(this);
        if (v2 != null) v2.removeConstraint(this);
        direction = NONE;
    }
}

class BMScaleConstraint extends BMBinaryConstraint {
    public var scale:BMVariable;
    public var offset:BMVariable;

    public function new(src:BMVariable, sc:BMVariable, off:BMVariable, dest:BMVariable, s:BMStrength) {
        scale = sc;
        offset = off;
        super(src, dest, s);
    }

    override public function addToGraph():Void {
        super.addToGraph();
        scale.addConstraint(this);
        offset.addConstraint(this);
    }

    override public function removeFromGraph():Void {
        super.removeFromGraph();
        if (scale != null) scale.removeConstraint(this);
        if (offset != null) offset.removeConstraint(this);
    }

    override public function markInputs(mark:Int):Void {
        super.markInputs(mark);
        scale.mark = mark;
        offset.mark = mark;
    }

    override public function execute():Void {
        if (direction == BMBinaryConstraint.FORWARD) {
            v2.value = v1.value * scale.value + offset.value;
        } else {
            v1.value = Std.int((v2.value - offset.value) / scale.value);
        }
    }

    override public function recalculate():Void {
        var ihn = input();
        var out = output();
        out.walkStrength = BMStrength.weakest(strength, ihn.walkStrength);
        out.stay = ihn.stay && scale.stay && offset.stay;
        if (out.stay) execute();
    }
}

class BMEqualityConstraint extends BMBinaryConstraint {
    public function new(a:BMVariable, b:BMVariable, s:BMStrength) {
        super(a, b, s);
    }

    override public function execute():Void {
        output().value = input().value;
    }
}

class BMPlan {
    public var list:Array<BMConstraint>;

    public function new() {
        list = new Array<BMConstraint>();
    }

    public function addConstraint(c:BMConstraint):Void {
        list.push(c);
    }

    public function size():Int { return list.length; }

    public function execute():Void {
        for (i in 0...list.length) {
            list[i].execute();
        }
    }
}

class BMPlanner {
    public var currentMark:Int;

    public function new() {
        currentMark = 0;
    }

    public function incrementalAdd(c:BMConstraint):Void {
        var mark = newMark();
        var overridden = c.satisfy(mark);
        while (overridden != null) {
            overridden = overridden.satisfy(mark);
        }
    }

    public function incrementalRemove(c:BMConstraint):Void {
        var out = c.output();
        c.markUnsatisfied();
        c.removeFromGraph();
        var unsatisfied = removePropagateFrom(out);
        var strength = BMStrength.REQUIRED;
        while (strength != BMStrength.WEAKEST) {
            for (i in 0...unsatisfied.length) {
                var u = unsatisfied[i];
                if (u.strength == strength) incrementalAdd(u);
            }
            strength = strength.nextWeaker();
        }
    }

    public function newMark():Int {
        currentMark = currentMark + 1;
        return currentMark;
    }

    public function makePlan(sources:Array<BMConstraint>):BMPlan {
        var mark = newMark();
        var plan = new BMPlan();
        var todo = new Array<BMConstraint>();
        for (i in 0...sources.length) {
            todo.push(sources[i]);
        }
        while (todo.length > 0) {
            var c = todo.pop();
            if (c.output().mark != mark && c.inputsKnown(mark)) {
                plan.addConstraint(c);
                c.output().mark = mark;
                addConstraintsConsumingTo(c.output(), todo);
            }
        }
        return plan;
    }

    public function extractPlanFromConstraints(constraints:Array<BMConstraint>):BMPlan {
        var sources = new Array<BMConstraint>();
        for (i in 0...constraints.length) {
            var c = constraints[i];
            if (c.isInput() && c.isSatisfied()) sources.push(c);
        }
        return makePlan(sources);
    }

    public function addPropagate(c:BMConstraint, mark:Int):Bool {
        var todo = new Array<BMConstraint>();
        todo.push(c);
        while (todo.length > 0) {
            var d = todo.pop();
            if (d.output().mark == mark) {
                incrementalRemove(c);
                return false;
            }
            d.recalculate();
            addConstraintsConsumingTo(d.output(), todo);
        }
        return true;
    }

    public function removePropagateFrom(out:BMVariable):Array<BMConstraint> {
        out.determinedBy = null;
        out.walkStrength = BMStrength.WEAKEST;
        out.stay = true;
        var unsatisfied = new Array<BMConstraint>();
        var todo = new Array<BMVariable>();
        todo.push(out);
        while (todo.length > 0) {
            var v = todo.pop();
            for (i in 0...v.constraints.length) {
                var c = v.constraints[i];
                if (!c.isSatisfied()) unsatisfied.push(c);
            }
            var determining = v.determinedBy;
            for (i in 0...v.constraints.length) {
                var next = v.constraints[i];
                if (next != determining && next.isSatisfied()) {
                    next.recalculate();
                    todo.push(next.output());
                }
            }
        }
        return unsatisfied;
    }

    public function addConstraintsConsumingTo(v:BMVariable, coll:Array<BMConstraint>):Void {
        var determining = v.determinedBy;
        for (i in 0...v.constraints.length) {
            var c = v.constraints[i];
            if (c != determining && c.isSatisfied()) coll.push(c);
        }
    }
}

class BMDeltaBlueCode {
    public static var planner:BMPlanner;
    public static var total:Int = 0;

    public static function chainTest(n:Int):Void {
        planner = new BMPlanner();
        var prev:BMVariable = null;
        var first:BMVariable = null;
        var last:BMVariable = null;
        for (i in 0...n + 1) {
            var v = new BMVariable("v", 0);
            if (prev != null) new BMEqualityConstraint(prev, v, BMStrength.REQUIRED);
            if (i == 0) first = v;
            if (i == n) last = v;
            prev = v;
        }
        new BMStayConstraint(last, BMStrength.STRONG_DEFAULT);
        var edit = new BMEditConstraint(first, BMStrength.PREFERRED);
        var edits = new Array<BMConstraint>();
        edits.push(edit);
        var plan = planner.extractPlanFromConstraints(edits);
        for (i in 0...100) {
            first.value = i;
            plan.execute();
            total = total + last.value;
        }
    }

    public static function projectionTest(n:Int):Void {
        planner = new BMPlanner();
        var scale = new BMVariable("scale", 10);
        var offset = new BMVariable("offset", 1000);
        var src:BMVariable = null;
        var dst:BMVariable = null;
        var dests = new Array<BMVariable>();
        for (i in 0...n) {
            src = new BMVariable("src", i);
            dst = new BMVariable("dst", i);
            dests.push(dst);
            new BMStayConstraint(src, BMStrength.NORMAL);
            new BMScaleConstraint(src, scale, offset, dst, BMStrength.REQUIRED);
        }
        change(src, 17);
        total = total + dst.value;
        if (dst.value != 1170) trace("Projection 1 failed");
        change(dst, 1050);
        total = total + src.value;
        if (src.value != 5) trace("Projection 2 failed");
        change(scale, 5);
        for (i in 0...n - 1) {
            total = total + dests[i].value;
            if (dests[i].value != i * 5 + 1000) trace("Projection 3 failed");
        }
        change(offset, 2000);
        for (i in 0...n - 1) {
            total = total + dests[i].value;
            if (dests[i].value != i * 5 + 2000) trace("Projection 4 failed");
        }
    }

    public static function change(v:BMVariable, newValue:Int):Void {
        var edit = new BMEditConstraint(v, BMStrength.PREFERRED);
        var edits = new Array<BMConstraint>();
        edits.push(edit);
        var plan = planner.extractPlanFromConstraints(edits);
        for (i in 0...10) {
            v.value = newValue;
            plan.execute();
        }
        edit.destroyConstraint();
    }

    public static function main():Void {
        for (i in 0...40) {
            chainTest(100);
            projectionTest(100);
        }
        trace("total: " + total);
    }
}
