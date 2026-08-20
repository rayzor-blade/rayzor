#!/usr/bin/env bash
# Move-checker oracle. ERROR = compilation must fail with a move diagnostic.
# SILENT = must compile with no move diagnostic at all.
#
# Single-file cases are `c*.hx`. A case that needs more than one file is a
# directory holding its own rayzor.toml, because a class only reaches MIR
# lowering in an importing compilation when the project declares a class-path.
cd "$(dirname "${BASH_SOURCE[0]}")"
R=/Users/amaterasu/Vibranium/rayzor/target/release/rayzor
expect_for() {
  case "$1" in
    c1_double_move)   echo ERROR ;;
    c2_return_branch) echo SILENT ;;
    c3_loop)          echo ERROR ;;
    c4_reassign)      echo SILENT ;;
    c5_legacy)        echo SILENT ;;
    c6_cross_file)    echo ERROR ;;
    c7_field_read)    echo ERROR ;;
    c8_method_receiver) echo SILENT ;;
    c9_branch_join)   echo ERROR ;;
    c10_borrow_param) echo SILENT ;;
    c11_borrow_after_move) echo ERROR ;;
    c12_capture_after_move) echo ERROR ;;
    c13_capture_before_move) echo SILENT ;;
    c14_cross_file_borrow) echo SILENT ;;
    *)                echo UNKNOWN ;;
  esac
}
classify() {
  if echo "$1" | grep -qaE "Error: \[E03(82|00)\]"; then echo ERROR
  elif echo "$1" | grep -qaE "Warning: \[E03(82|00)\]"; then echo WARN
  else echo SILENT; fi
}
score() {
  local n="$1" got="$2" want
  want="$(expect_for "$n")"
  if [ "$got" = "$want" ]; then printf "  PASS  %-20s %s\n" "$n" "$got"; pass=$((pass+1))
  else printf "  FAIL  %-20s want=%-7s got=%s\n" "$n" "$want" "$got"; fail=$((fail+1)); fi
}
pass=0; fail=0
# A warm cache skips MIR lowering, and with it the analysis, so every case
# starts cold or the run scores the cache rather than the compiler.
rm -rf .rayzor
for f in c*.hx; do
  n="${f%.hx}"
  score "$n" "$(classify "$($R run "$f" --release 2>&1 | sed 's/\x1b\[[0-9;]*m//g')")"
done
for d in c*/; do
  n="${d%/}"
  [ -f "$d/rayzor.toml" ] || continue
  rm -rf "$d/.rayzor"
  score "$n" "$(classify "$(cd "$d" && $R run Main.hx --release 2>&1 | sed 's/\x1b\[[0-9;]*m//g')")"
done
echo "  ---- $pass passed, $fail failed"
