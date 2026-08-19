#!/usr/bin/env bash
# The comparable block of a captured run: plan, census, cache and generated text.
#
# Masked because two identical runs move them: per-worker timings and claim
# counts are chunk-stealing artefacts, so only their sum is deterministic. The
# gate line is COUNTED rather than compared — its print is a first-read side
# effect, so relocating the first read moves the line without changing routing.
f="$1"
{ grep -a '^\[nue-plan\]\|^\[nue-graph\]\|^\[q4-census\]\|^\[kv-cache\]\|^\[lm_head\]' "$f"
  grep -ao 'dispatches=[0-9]*' "$f"
  grep -ac '^\[q4-gate\]' "$f" | sed 's/^/q4gate_lines=/'
  sed -n '/^\[output\] /,$p' "$f"
} | sed -E 's/(band_ms|quant_ms)=[0-9.]+//g; s/w[0-9]+=[0-9.]+ms\/[0-9]+c//g'
