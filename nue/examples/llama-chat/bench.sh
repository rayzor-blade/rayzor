#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RAYZOR="${RAYZOR:-../../../target/release/rayzor}"
GGUF="${GGUF:-/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf}"
PROMPT="${PROMPT:-Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation}"
MAX_TOKENS="${MAX_TOKENS:-5000}"
TEMP="${TEMP:-0.7}"
RUNS="${RUNS:-6}"
TIMEOUT="${TIMEOUT:-240}"
COOLDOWN_MS="${COOLDOWN_MS:-15000}"
START_INTERPRETED="${START_INTERPRETED:-true}"
TIER_PROMOTION="${TIER_PROMOTION:-true}"
VARIANTS="${VARIANTS:-1/20/5 1/20/2 1/30/5}"
DECODE_PROFILE="${DECODE_PROFILE:-false}"

read -r -a VARIANT_LIST <<< "$VARIANTS"

cmd=(
  "$RAYZOR" debug bench Main.hx
  --metric tok-per-s
  -n "$RUNS"
  --timeout "$TIMEOUT"
  --tier-start-interpreted "$START_INTERPRETED"
  --tier-promotion "$TIER_PROMOTION"
)

for variant in "${VARIANT_LIST[@]}"; do
  cmd+=(--tier-thresholds "$variant")
done

if [[ "$DECODE_PROFILE" == "true" || "$DECODE_PROFILE" == "1" || "$DECODE_PROFILE" == "yes" ]]; then
  cmd+=(--decode-profile)
fi

cmd+=(
  --cooldown-ms "$COOLDOWN_MS"
  -- "$GGUF" "$PROMPT" "$MAX_TOKENS" "$TEMP"
)

"${cmd[@]}"
