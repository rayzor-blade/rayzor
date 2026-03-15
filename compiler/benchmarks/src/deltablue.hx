// DeltaBlue Constraint Solver Benchmark
// Ported from Dart (originally Smalltalk → JavaScript → Dart by Google 2008-2010)
//
// "The DeltaBlue Algorithm: An Incremental Constraint Hierarchy Solver"
//   Bjorn N. Freeman-Benson and John Maloney, January 1990
//
// Tests: OOP inheritance, polymorphic dispatch, constraint propagation, arrays

package benchmarks;

@:safety(false)
class Strength {
    public static var REQUIRED:Strength;
    public static var STRONG_PREFERRED:Strength;
    public static var PREFERRED:Strength;
    public static var STRONG_DEFAULT:Strength;
    public static var NORMAL:Strength;
    public static var WEAK_DEFAULT:Strength;
    public static var WEAKEST:Strength;

    public var value:Int;
    public var name:String;

    public function new(v:Int, n:String) {
        value = v;
        name = n;
    }

    public static function init():Void {
        REQUIRED = new Strength(0, "required");
        STRONG_PREFERRED = new Strength(1, "strongPreferred");
        PREFERRED = new Strength(2, "preferred");
        STRONG_DEFAULT = new Strength(3, "strongDefault");
        NORMAL = new Strength(4, "normal");
        WEAK_DEFAULT = new Strength(5, "weakDefault");
        WEAKEST = new Strength(6, "weakest");
    }

    public function nextWeaker():Strength {
        if (value == 0) return WEAKEST;
        if (value == 1) return WEAK_DEFAULT;
        if (value == 2) return NORMAL;
        if (value == 3) return STRONG_DEFAULT;
        if (value == 4) return PREFERRED;
        if (value == 5) return STRONG_PREFERRED;
        return WEAKEST;
    }

    public static function stronger(s1:Strength, s2:Strength):Bool {
        return s1.value < s2.value;
    }

    public static function weaker(s1:Strength, s2:Strength):Bool {
        return s1.value > s2.value;
    }

    public static function weakest(s1:Strength, s2:Strength):Strength {
        return weaker(s1, s2) ? s1 : s2;
    }

    public static function strongest(s1:Strength, s2:Strength):Strength {
        return stronger(s1, s2) ? s1 : s2;
    }
}
@:safety(false)
class Variable {
    public var constraints:Array<Constraint>;
    public var determinedBy:Constraint;
    public var mark:Int;
    public var walkStrength:Strength;
    public var stay:Bool;
    public var value:Int;
    public var name:String;

    public function new(n:String, v:Int) {
        name = n;
        value = v;
        constraints = new Array<Constraint>();
        determinedBy = null;
        mark = 0;
        walkStrength = Strength.WEAKEST;
        stay = true;
    }

    public function addConstraint(c:Constraint):Void {
        constraints.push(c);
    }

    public function removeConstraint(c:Constraint):Void {
        var newConstraints = new Array<Constraint>();
        for (i in 0...constraints.length) {
            if (constraints[i] != c) {
                newConstraints.push(constraints[i]);
            }
        }
        constraints = newConstraints;
        if (determinedBy == c) determinedBy = null;
    }
}
@:safety(false)
class Constraint {
    public var strength:Strength;

    public function new(s:Strength) {
        strength = s;
    }

    public function isSatisfied():Bool { return false; }
    public function markUnsatisfied():Void {}
    public function addToGraph():Void {}
    public function removeFromGraph():Void {}
    public function chooseMethod(mark:Int):Void {}
    public function markInputs(mark:Int):Void {}
    public function inputsKnown(mark:Int):Bool { return false; }
    public function output():Variable { return null; }
    public function execute():Void {}
    public function recalculate():Void {}
    public function isInput():Bool { return false; }

    public function addConstraint():Void {
        addToGraph();
        DeltaBlue.planner.incrementalAdd(this);
    }

    public function satisfy(mark:Int):Constraint {
        chooseMethod(mark);
        if (!isSatisfied()) {
            if (strength == Strength.REQUIRED) {
                trace("Could not satisfy a required constraint!");
            }
            return null;
        }
        markInputs(mark);
        var out = output();
        var overridden = out.determinedBy;
        if (overridden != null) overridden.markUnsatisfied();
        out.determinedBy = this;
        if (!DeltaBlue.planner.addPropagate(this, mark)) trace("Cycle encountered");
        out.mark = mark;
        return overridden;
    }

    public function destroyConstraint():Void {
        if (isSatisfied()) DeltaBlue.planner.incrementalRemove(this);
        removeFromGraph();
    }
}
@:safety(false)
class UnaryConstraint extends Constraint {
    public var myOutput:Variable;
    public var satisfied:Bool;

    public function new(output:Variable, s:Strength) {
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
        satisfied = (myOutput.mark != mark) && Strength.stronger(strength, myOutput.walkStrength);
    }

    override public function isSatisfied():Bool { return satisfied; }
    override public function markInputs(mark:Int):Void {}
    override public function output():Variable { return myOutput; }

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
@:safety(false)
class StayConstraint extends UnaryConstraint {
    public function new(v:Variable, str:Strength) {
        super(v, str);
    }

    override public function execute():Void {}
}
@:safety(false)
class EditConstraint extends UnaryConstraint {
    public function new(v:Variable, str:Strength) {
        super(v, str);
    }

    override public function isInput():Bool { return true; }
    override public function execute():Void {}
}
@:safety(false)
class BinaryConstraint extends Constraint {
    public var v1:Variable;
    public var v2:Variable;
    public var direction:Int;

    public static inline var NONE = 1;
    public static inline var FORWARD = 2;
    public static inline var BACKWARD = 0;

    public function new(a:Variable, b:Variable, s:Strength) {
        super(s);
        v1 = a;
        v2 = b;
        direction = NONE;
        addConstraint();
    }

    override public function chooseMethod(mark:Int):Void {
        if (v1.mark == mark) {
            var d = (v2.mark != mark && Strength.stronger(strength, v2.walkStrength))
                ? FORWARD : NONE;
            direction = d;
        } else if (v2.mark == mark) {
            var d = (v1.mark != mark && Strength.stronger(strength, v1.walkStrength))
                ? BACKWARD : NONE;
            direction = d;
        } else if (Strength.weaker(v1.walkStrength, v2.walkStrength)) {
            var d = Strength.stronger(strength, v1.walkStrength)
                ? BACKWARD : NONE;
            direction = d;
        } else {
            var d = Strength.stronger(strength, v2.walkStrength)
                ? FORWARD : BACKWARD;
            direction = d;
        }
    }

    override public function addToGraph():Void {
        v1.addConstraint(this);
        v2.addConstraint(this);
        direction = NONE;
    }

    override public function isSatisfied():Bool { return direction != NONE; }

    override public function markInputs(mark:Int):Void {
        var inp = input();
        inp.mark = mark;
    }

    public function input():Variable {
        return direction == BinaryConstraint.FORWARD ? v1 : v2;
    }

    override public function output():Variable {
        return direction == BinaryConstraint.FORWARD ? v2 : v1;
    }

    override public function recalculate():Void {
        var ihn = input();
        var out = output();
        out.walkStrength = Strength.weakest(strength, ihn.walkStrength);
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
@:safety(false)
class ScaleConstraint extends BinaryConstraint {
    public var scale:Variable;
    public var offset:Variable;

    public function new(src:Variable, sc:Variable, off:Variable, dest:Variable, s:Strength) {
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
        if (direction == BinaryConstraint.FORWARD) {
            v2.value = v1.value * scale.value + offset.value;
        } else {
            v1.value = Std.int((v2.value - offset.value) / scale.value);
        }
    }

    override public function recalculate():Void {
        var ihn = input();
        var out = output();
        out.walkStrength = Strength.weakest(strength, ihn.walkStrength);
        out.stay = ihn.stay && scale.stay && offset.stay;
        if (out.stay) execute();
    }
}
@:safety(false)
class EqualityConstraint extends BinaryConstraint {
    public function new(a:Variable, b:Variable, s:Strength) {
        super(a, b, s);
    }

    override public function execute():Void {
        var out = output();
        var inp = input();
        out.value = inp.value;
    }
}
@:safety(false)
class Plan {
    public var list:Array<Constraint>;

    public function new() {
        list = new Array<Constraint>();
    }

    public function addConstraint(c:Constraint):Void {
        list.push(c);
    }

    public function size():Int { return list.length; }

    public function execute():Void {
        for (i in 0...list.length) {
            var c = list[i];
            c.execute();
        }
    }
}
@:safety(false)
class Planner {
    public var currentMark:Int;

    public function new() {
        currentMark = 0;
    }

    public function incrementalAdd(c:Constraint):Void {
        var mark = newMark();
        var overridden = c.satisfy(mark);
        while (overridden != null) {
            overridden = overridden.satisfy(mark);
        }
    }

    public function incrementalRemove(c:Constraint):Void {
        var out = c.output();
        c.markUnsatisfied();
        c.removeFromGraph();
        var unsatisfied = removePropagateFrom(out);
        var strength = Strength.REQUIRED;
        while (strength != Strength.WEAKEST) {
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

    public function makePlan(sources:Array<Constraint>):Plan {
        var mark = newMark();
        var plan = new Plan();
        var todo = new Array<Constraint>();
        for (i in 0...sources.length) {
            todo.push(sources[i]);
        }
        while (todo.length > 0) {
            var c = todo.pop();
            var out = c.output();
            if (out.mark != mark && c.inputsKnown(mark)) {
                plan.addConstraint(c);
                out.mark = mark;
                addConstraintsConsumingTo(out, todo);
            }
        }
        return plan;
    }

    public function extractPlanFromConstraints(constraints:Array<Constraint>):Plan {
        var sources = new Array<Constraint>();
        for (i in 0...constraints.length) {
            var c = constraints[i];
            if (c.isInput() && c.isSatisfied()) sources.push(c);
        }
        return makePlan(sources);
    }

    public function addPropagate(c:Constraint, mark:Int):Bool {
        var todo = new Array<Constraint>();
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

    public function removePropagateFrom(out:Variable):Array<Constraint> {
        out.determinedBy = null;
        out.walkStrength = Strength.WEAKEST;
        out.stay = true;
        var unsatisfied = new Array<Constraint>();
        var todo = new Array<Variable>();
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

    public function addConstraintsConsumingTo(v:Variable, coll:Array<Constraint>):Void {
        var determining = v.determinedBy;
        for (i in 0...v.constraints.length) {
            var c = v.constraints[i];
            if (c != determining && c.isSatisfied()) {
                coll.push(c);
            }
        }
    }
}
@:safety(false)
class DeltaBlue {
    public static var planner:Planner;
    public static var total:Int = 0;

    public static function chainTest(n:Int):Void {
        planner = new Planner();
        var prev:Variable = null;
        var first:Variable = null;
        var last:Variable = null;
        for (i in 0...n + 1) {
            var v = new Variable("v", 0);
            if (prev != null) new EqualityConstraint(prev, v, Strength.REQUIRED);
            if (i == 0) first = v;
            if (i == n) last = v;
            prev = v;
        }
        new StayConstraint(last, Strength.STRONG_DEFAULT);
        var edit = new EditConstraint(first, Strength.PREFERRED);
        var edits = new Array<Constraint>();
        edits.push(edit);
        var plan = planner.extractPlanFromConstraints(edits);
        for (i in 0...100) {
            first.value = i;
            plan.execute();
            total = total + last.value;
        }
    }

    public static function projectionTest(n:Int):Void {
        planner = new Planner();
        var scale = new Variable("scale", 10);
        var offset = new Variable("offset", 1000);
        var src:Variable = null;
        var dst:Variable = null;
        var dests = new Array<Variable>();
        for (i in 0...n) {
            src = new Variable("src", i);
            dst = new Variable("dst", i);
            dests.push(dst);
            new StayConstraint(src, Strength.NORMAL);
            new ScaleConstraint(src, scale, offset, dst, Strength.REQUIRED);
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

    public static function change(v:Variable, newValue:Int):Void {
        var edit = new EditConstraint(v, Strength.PREFERRED);
        var edits = new Array<Constraint>();
        edits.push(edit);
        var plan = planner.extractPlanFromConstraints(edits);
        for (i in 0...10) {
            v.value = newValue;
            plan.execute();
        }
        edit.destroyConstraint();
    }

    public static function main():Void {
        Strength.init();
        chainTest(100);
        projectionTest(100);
        trace("total: " + total);
    }
}
