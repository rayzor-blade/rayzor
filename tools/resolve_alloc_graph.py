#!/usr/bin/env python3
"""Resolve PCs in /tmp/rayzor_alloc_graph.csv against:
  * /tmp/rayzor_jit_symbols.csv (RAYZOR_DUMP_JIT_MAP=1) — JIT fn name +
    file_id + line per address range
  * /tmp/rayzor_file_table.csv  (RAYZOR_DUMP_JIT_MAP=1) — file_id → path
  * the rayzor binary via `atos -o BIN -arch arm64` for PCs in the
    binary's __TEXT segment

Output: /tmp/rayzor_alloc_attributed.csv with one row per top alloc site:
    rank, sampled_allocs, est_total_allocs, name1..name6
where each name is either:
    "qname (path:line)"  — JIT-attributed Haxe source
    "<atos_symbol>"      — Rust binary symbol
    "?(0xPC)"            — neither resolved

Also prints a per-Haxe-file aggregate of est_total_allocs at the end.

Usage:
  tools/resolve_alloc_graph.py [path/to/rayzor]
"""

import bisect
import csv
import os
import subprocess
import sys

GRAPH_CSV   = "/tmp/rayzor_alloc_graph.csv"
JIT_CSV     = "/tmp/rayzor_jit_symbols.csv"
FILE_TABLE  = "/tmp/rayzor_file_table.csv"
OUT_CSV     = "/tmp/rayzor_alloc_attributed.csv"
DEFAULT_BIN = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "target/release/rayzor",
)

def load_file_table(path):
    """file_id → path."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                out[int(r["file_id"])] = r["path"]
            except (ValueError, KeyError):
                continue
    return out

def load_jit_map(path, file_table):
    """Returns sorted list of (start, end, qname_with_loc)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                start = int(r["start_hex"], 16)
                end   = int(r["end_hex"], 16)
            except (ValueError, KeyError):
                continue
            if end <= start:
                end = start + 1024
            qname = r.get("qname", "?")
            try:
                file_id = int(r.get("file_id", "0"))
                line    = int(r.get("line", "0"))
            except ValueError:
                file_id, line = 0, 0
            if file_id > 0 and file_id in file_table and line > 0:
                short = os.path.basename(file_table[file_id])
                label = f"{qname} ({short}:{line})"
            else:
                label = qname
            rows.append((start, end, label, qname))
    rows.sort(key=lambda r: r[0])
    return rows

def resolve_jit(pc, jit_rows):
    starts = [r[0] for r in jit_rows]
    idx = bisect.bisect_right(starts, pc) - 1
    if idx < 0:
        return None, None
    start, end, label, qname = jit_rows[idx]
    if pc < end:
        return label, qname
    return None, None

def resolve_atos(pcs, binary):
    if not pcs:
        return {}
    cmd = ["atos", "-o", binary, "-arch", "arm64"] + [f"0x{pc:x}" for pc in pcs]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except Exception:
        return {pc: None for pc in pcs}
    if out.returncode != 0:
        return {pc: None for pc in pcs}
    syms = out.stdout.strip().split("\n")
    if len(syms) < len(pcs):
        syms += [""] * (len(pcs) - len(syms))
    return {pc: (s if s and not s.startswith("0x") else None) for pc, s in zip(pcs, syms)}

def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BIN
    if not os.path.exists(GRAPH_CSV):
        print(f"missing {GRAPH_CSV} — run with RAYZOR_ALLOC_GRAPH=1", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(JIT_CSV):
        print(f"missing {JIT_CSV} — run with RAYZOR_DUMP_JIT_MAP=1", file=sys.stderr)
        sys.exit(1)

    file_table = load_file_table(FILE_TABLE)
    jit_rows   = load_jit_map(JIT_CSV, file_table)
    print(f"loaded {len(file_table)} files, {len(jit_rows)} JIT symbols", file=sys.stderr)

    rows_in = list(csv.DictReader(open(GRAPH_CSV)))

    binary_pcs = set()
    for r in rows_in:
        for k in ("pc1", "pc2", "pc3", "pc4", "pc5", "pc6"):
            try:
                pc = int(r[k], 16)
            except (ValueError, KeyError):
                continue
            if pc == 0:
                continue
            label, _ = resolve_jit(pc, jit_rows)
            if label is None:
                binary_pcs.add(pc)

    atos_map = resolve_atos(sorted(binary_pcs), binary) if binary and os.path.exists(binary) else {}
    if atos_map:
        n_atos = sum(1 for v in atos_map.values() if v)
        print(f"atos resolved {n_atos}/{len(atos_map)} binary PCs", file=sys.stderr)

    bucket_counts = {"jit": 0, "atos": 0, "unknown": 0}
    site_attribution = {}   # first_resolved_label -> est_total_allocs
    haxe_file_bytes  = {}   # file_path -> est_total_allocs
    with open(OUT_CSV, "w") as fout:
        w = csv.writer(fout)
        w.writerow(["rank", "sampled_allocs", "est_total_allocs", "name1", "name2", "name3", "name4", "name5", "name6"])
        for r in rows_in:
            est = int(r["est_total_allocs"])
            names = []
            first_label = None
            first_qname = None
            for k in ("pc1", "pc2", "pc3", "pc4", "pc5", "pc6"):
                try:
                    pc = int(r[k], 16)
                except (ValueError, KeyError):
                    names.append("")
                    continue
                if pc == 0:
                    names.append("")
                    continue
                label, qname = resolve_jit(pc, jit_rows)
                if label is not None:
                    names.append(label)
                    bucket_counts["jit"] += 1
                    if first_label is None:
                        first_label = label
                        first_qname = qname
                elif atos_map.get(pc):
                    names.append(atos_map[pc])
                    bucket_counts["atos"] += 1
                    if first_label is None:
                        first_label = atos_map[pc]
                else:
                    names.append(f"?(0x{pc:x})")
                    bucket_counts["unknown"] += 1
            if first_label is None:
                first_label = names[0] if names else "<empty>"
            w.writerow([r["rank"], r["sampled_allocs"], r["est_total_allocs"]] + names)
            site_attribution[first_label] = site_attribution.get(first_label, 0) + est
            # Haxe-file rollup: pick the file from the first JIT frame
            if first_qname:
                # Walk all frames in the row, find first JIT entry that has a known file
                for k in ("pc1", "pc2", "pc3", "pc4", "pc5", "pc6"):
                    try:
                        pc = int(r[k], 16)
                    except (ValueError, KeyError):
                        continue
                    if pc == 0:
                        continue
                    starts = [x[0] for x in jit_rows]
                    idx = bisect.bisect_right(starts, pc) - 1
                    if idx < 0:
                        continue
                    s, e, _, _ = jit_rows[idx]
                    if pc < e:
                        # find file via jit row's file_id; reload from CSV
                        break
                # Simpler: just match based on label text
                # Label is "qname (basename:line)" — extract basename
                if "(" in first_label and ":" in first_label:
                    fn_basename = first_label.rsplit("(", 1)[1].rstrip(")").split(":")[0]
                    haxe_file_bytes[fn_basename] = haxe_file_bytes.get(fn_basename, 0) + est

    print(f"wrote {OUT_CSV}", file=sys.stderr)
    print(f"frame attribution: {bucket_counts}", file=sys.stderr)

    print("\n=== top 25 sites by est_total_allocs ===")
    for name, est in sorted(site_attribution.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {est:>12,}  {name}")

    print("\n=== alloc volume by Haxe file ===")
    for path, est in sorted(haxe_file_bytes.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {est:>12,}  {path}")

if __name__ == "__main__":
    main()
