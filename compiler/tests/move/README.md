# Move-checker oracle

What the ownership checker must do, written as cases rather than prose. Each
file declares its expectation in `check.sh`:

- `ERROR`  — compilation must fail with an ownership diagnostic (E0382 use
  after move, E0383 escaping borrow)
- `SILENT` — must compile with no ownership diagnostic at all

Run with `./check.sh` (needs `target/release/rayzor` built).

These are deliberately kept out of `compiler/tests/haxe/`, which the main suite
globs and scores by exit code — these cases are about whether a diagnostic
fires, not about what the program prints.

| case | expects | why it matters |
| --- | --- | --- |
| `c1_double_move` | ERROR | the same value passed to two calls |
| `c2_return_branch` | SILENT | the move is on a path that returns, so the later read is unreachable |
| `c3_loop` | ERROR | moved in a loop body, still moved on the next iteration |
| `c4_reassign` | SILENT | reassignment revives the binding |
| `c5_legacy` | SILENT | no `@:safety` anywhere — legacy Haxe must never be analysed |
| `c6_cross_file` | ERROR | the `@:move` class is reached through an import |
| `c7_field_read` | ERROR | reading a field of a moved binding is still a use |
| `c8_method_receiver` | SILENT | `a.f()` observes the receiver, it does not consume it |
| `c9_branch_join` | ERROR | moved on one branch, read after the two rejoin |
| `c10_borrow_param` | SILENT | the parameter says `@:borrow`, so the call does not consume |
| `c11_borrow_after_move` | ERROR | a borrow of an already-moved value is still a use |
| `c12_capture_after_move` | ERROR | a closure captures a binding that was moved |
| `c13_capture_before_move` | SILENT | captured while still live |
| `c14_cross_file_borrow` | SILENT | the `@:borrow` is declared in another module |
| `c15_consume_receiver` | ERROR | `@:consume` ends the receiver, so the next call is a use after move |
| `c16_consume_last` | SILENT | nothing follows the consuming call |
| `c17_consume_borrowed_param` | SILENT | a consumed receiver shifts the declared parameters by one |
| `c18_cross_file_consume` | ERROR | the `@:consume` is declared in another module |
| `c19_void_call` | ERROR | a call in statement position, whose helper returns `Void` |
| `c20_void_consume` | ERROR | `s.close();` — how a consuming call is actually written |
| `c21_borrow_returned` | ERROR | a borrow handed back outlives the call it was lent for |
| `c22_borrow_returned_alias` | ERROR | the same, laundered through a local |
| `c23_borrow_field_read` | SILENT | `return r.v` reads a field; it does not return the borrow |
| `c24_borrow_stored` | ERROR | a borrow written into a field |
| `c25_borrow_captured` | ERROR | a borrow captured by a closure |
| `c26_borrow_laundered_ctor` | ERROR | a borrow wrapped in a constructor and handed back |
| `c27_borrow_laundered_local` | ERROR | the same, bound to a local first |
| `c28_borrow_arg_passthrough` | SILENT | returning a call's *result* is not returning the borrow |

`c2`, `c4` and `c8` need branch, reassignment and receiver awareness to stay
silent for the right reason; a checker that detects nothing also passes them.
The silent cases are what stop the checker being made to fire by making it
fire on everything.

`c6`, `c14` and `c18` are directories rather than files: an imported declaration only
reaches MIR lowering in the importing compilation when the project declares a
class-path, so those cases carry their own `rayzor.toml`. Both have a control —
removing the annotation must flip the result — because a case that is silent
for the wrong reason passes just as well as one that is silent for the right
one.

A helper's RETURN TYPE is part of the case. A void call lowers to no
register, so a checker can be blind to `f(a);` while catching `var x = f(a);`.
Cases 1-18 all returned `Int` and 18/18 passed while exactly that hole was
open; `c19`/`c20` exist because a green oracle is only as good as the shapes
it contains.

Every case runs cold. A warm cache skips MIR lowering, and with it the
analysis, so a second run of a failing case would score the cache.
