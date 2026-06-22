// Regression: cross-module transitive monomorphization of a generic method chain.
//
// haxe.ds.BalancedTree<K,V> is compiled as an imported stdlib module and merged
// into the user program. Its public API (set/get) drives a chain of generic
// methods — set -> setLoop -> balance/compare -> new TreeNode<K,V> — that all
// inherit the class type params. When the class is IMPORTED (not in the same
// file), the entry call `t.set(1,10)` resolves to a cross-module function that
// is not yet in the module at lowering time, so the callee-based type_args
// inference saw nothing and attached no type_args. The monomorphizer then could
// not seed the specialization, the whole chain reached codegen as generic trap
// stubs, and `t.set(...)` SIGILL'd (exit 132).
//
// Fixed in hir_to_mir's method-call lowering: when the receiver is a concrete
// generic-class instance (Class/GenericInstance carrying type_args), attach the
// receiver's type_args even when the cross-module callee is invisible. That seeds
// the monomorphizer, whose existing transitive propagation (propagate_transitive_
// fixups) then specializes the rest of the chain via the caller's substitution
// map. Verified bit-for-bit for both Int keys and String keys (Reflect.compare).

class Main {
    static function main() {
        // Int keys: exercises the integer compare path. Three inserts force a
        // tree rotation through balance(). Sum the lookups (arithmetic coerces
        // Null<Int>) rather than `==`, which on Null<Int> is unreliable.
        var ti = new haxe.ds.BalancedTree<Int, Int>();
        ti.set(1, 10);
        ti.set(2, 20);
        ti.set(3, 30);
        var isum = ti.get(1) + ti.get(2) + ti.get(3); // expect 60

        // String keys: exercises Reflect.compare on strings (type tag must NOT
        // erase to Int, or the BST ordering/lookup would break).
        var ts = new haxe.ds.BalancedTree<String, Int>();
        ts.set("banana", 2);
        ts.set("apple", 1);
        ts.set("cherry", 3);
        var ssum = ts.get("apple") + ts.get("banana") + ts.get("cherry"); // expect 6

        Sys.println((isum == 60 && ssum == 6) ? "PASS" : "FAIL isum=" + isum + " ssum=" + ssum);
    }
}
