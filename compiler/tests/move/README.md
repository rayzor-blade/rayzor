# Move-checker oracle

What the ownership checker must do, written as cases rather than prose. Each
file declares its expectation in `check.sh`:

- `ERROR`  — compilation must fail with a move diagnostic
- `SILENT` — must compile with no move diagnostic at all

Run with `./check.sh` (needs `target/release/rayzor` built).

These are deliberately kept out of `compiler/tests/haxe/`, which the main suite
globs and scores by exit code. Two cases fail today; that is the point. They
record the gap rather than hide it.

| case | expects | why it matters |
| --- | --- | --- |
| `c1_double_move` | ERROR | the same value passed to two calls — missed entirely today |
| `c2_return_branch` | SILENT | the move is on a path that returns, so the later read is unreachable |
| `c3_loop` | ERROR | moved in a loop body, still moved on the next iteration |
| `c4_reassign` | SILENT | reassignment revives the binding |
| `c5_legacy` | SILENT | no `@:safety` anywhere — legacy Haxe must never be analysed |

`c2` and `c4` need branch and reassignment awareness to stay silent for the
right reason; a checker that detects nothing also passes them, which is why
`c1` and `c3` are the ones that measure progress.
