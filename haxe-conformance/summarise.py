#!/usr/bin/env python3
"""Turn a conformance report into a shields.io badge and a markdown summary.

Kept out of the workflow YAML so it can be run and corrected locally against a
report from any machine, rather than only by pushing and watching CI.
"""
import collections
import json
import pathlib
import sys

ORDER = ["PASS", "WRONG_ANSWER", "NO_OUTPUT", "COMPILE_FAIL", "CRASH", "TIMEOUT", "SKIP"]


def read(path):
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            # The suite column is newer than the report format; a report
            # written before it exists still reads, as one unnamed suite.
            rows.append((parts[0], parts[1], parts[2],
                         parts[3] if len(parts) >= 4 else "unit"))
    return rows


def colour(pct):
    # Bands, not a gradient: a badge that drifts from red to orange on a
    # fraction of a percent reads as progress that did not happen.
    for edge, name in ((90, "brightgreen"), (75, "green"), (50, "yellowgreen"),
                       (25, "yellow"), (10, "orange")):
        if pct >= edge:
            return name
    return "red"


def main():
    report = sys.argv[1]
    outdir = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read(report)
    counts = collections.Counter(status for _, status, _, _ in rows)
    # Skipped tests are out of scope, not failures: counting them would let the
    # score rise by skipping more.
    scored = sum(c for s, c in counts.items() if s != "SKIP")
    passed = counts.get("PASS", 0)
    pct = (100.0 * passed / scored) if scored else 0.0

    badge = {
        "schemaVersion": 1,
        "label": "haxe conformance",
        "message": f"{passed}/{scored} ({pct:.1f}%)",
        "color": colour(pct),
    }
    (outdir / "badge.json").write_text(json.dumps(badge) + "\n", encoding="utf-8")

    lines = [f"## Haxe conformance — {passed}/{scored} ({pct:.1f}%)", ""]
    # Per suite as well as overall. The language suite is two orders of
    # magnitude larger than sys and threads, so a combined figure moves only
    # with it and the other two can go to zero unnoticed.
    per = {}
    for _, status, _, suite in rows:
        per.setdefault(suite, collections.Counter())[status] += 1
    if len(per) > 1:
        lines += ["| suite | scored | passed | % | skipped |", "|---|---:|---:|---:|---:|"]
        for suite in sorted(per):
            c = per[suite]
            sc = sum(n for st, n in c.items() if st != "SKIP")
            pa = c.get("PASS", 0)
            pp = (100.0 * pa / sc) if sc else 0.0
            lines.append(f"| {suite} | {sc} | {pa} | {pp:.1f}% | {c.get('SKIP', 0)} |")
        lines.append("")

    lines += ["| outcome | count |", "|---|---:|"]
    for status in ORDER + sorted(set(counts) - set(ORDER)):
        if counts.get(status):
            lines.append(f"| {status} | {counts[status]} |")

    # What the compiler could not build, most frequent first. This is the part
    # that turns a failure count into a work list.
    stubs = collections.Counter(
        detail.split("uncompiled ", 1)[1].strip()
        for _, _, detail, _ in rows if detail.startswith("uncompiled ")
    )
    if stubs:
        shapes = collections.Counter(name.rsplit(".", 1)[-1] for name in stubs)
        lines += ["", "### Uncompiled functions", "",
                  f"{sum(stubs.values())} failures name a function the compiler "
                  f"could not build.", "", "| member | count |", "|---|---:|"]
        for shape, count in shapes.most_common(10):
            lines.append(f"| `.{shape}` | {count} |")

    summary = "\n".join(lines) + "\n"
    (outdir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
