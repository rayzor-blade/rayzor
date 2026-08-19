#!/usr/bin/env bash
# Move-checker oracle. ERROR = compilation must fail with a move diagnostic.
# SILENT = must compile with no move diagnostic at all.
cd "$(dirname "${BASH_SOURCE[0]}")"
R=/Users/amaterasu/Vibranium/rayzor/target/release/rayzor
expect_for() {
  case "$1" in
    c1_double_move)   echo ERROR ;;
    c2_return_branch) echo SILENT ;;
    c3_loop)          echo ERROR ;;
    c4_reassign)      echo SILENT ;;
    c5_legacy)        echo SILENT ;;
    *)                echo UNKNOWN ;;
  esac
}
pass=0; fail=0
for f in c*.hx; do
  n="${f%.hx}"; want="$(expect_for "$n")"
  out="$($R run "$f" --release 2>&1 | sed 's/\x1b\[[0-9;]*m//g')"
  if echo "$out" | grep -qaE "Error: \[E03(82|00)\]"; then got=ERROR
  elif echo "$out" | grep -qaE "Warning: \[E03(82|00)\]"; then got=WARN
  else got=SILENT; fi
  if [ "$got" = "$want" ]; then printf "  PASS  %-20s %s\n" "$n" "$got"; pass=$((pass+1))
  else printf "  FAIL  %-20s want=%-7s got=%s\n" "$n" "$want" "$got"; fail=$((fail+1)); fi
done
echo "  ---- $pass passed, $fail failed"
