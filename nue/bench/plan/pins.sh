#!/usr/bin/env bash
# Pin every gate that can move a dispatch count or a generated token.
#
# Scrub the aliases FIRST: Sys.getEnvOr falls back to the RAYZOR_*/RZT_* alias
# when the NUE_* name is unset, so a value left in the environment silently
# supplies a gate that this list means to pin.
for v in $(env | grep -oE '^(RAYZOR|RZT|NUE)_[A-Z0-9_]*'); do unset "$v"; done
export NUE_MATMUL=1 NUE_INT8=0 NUE_HAXE_INT8=1 NUE_HAXE_Q8_0=1 \
  NUE_FUSED_MATMUL=0 NUE_FUSED_ROWWISE=1 NUE_FUSED_DISPATCH=1 \
  NUE_NROW_BLOCK=64 NUE_AMX_MIN_BATCH=128 RZT_AMX_PREFILL=1 NUE_AMX_HAXE=1 \
  NUE_FLASH=1 NUE_KV_Q8=1 NUE_FLASH_POOL=1 NUE_FLASH_SHIFTED_Q=0 \
  NUE_FLASH_BATCH=1073741824 \
  NUE_REQUANT_LM_HEAD=1 NUE_REQUANT_Q6K=0 NUE_Q8_0_QUANT=1 \
  NUE_DECODE_WARM=1 NUE_FREE_GGUF_BYTES=1 \
  NUE_PREFILL=off NUE_PREFILL_LAST_LOGITS=0 \
  NUE_POOL_PROFILE=throughput NUE_MATMUL_WORKERS=8 NUE_POOL_ADAPTIVE=1 \
  NUE_DUMP_BLOCK_SHAPES=0 NUE_PROFILE_DECODE_SPLIT=0 NUE_PROFILE_ATTN=0 \
  NUE_DUMP_TOPK=0
# NUE_POOL_SPINS / NUE_POOL_RELAX stay UNSET — their defaults are derived, and
# a literal value would pin something the platform is meant to choose.
exec "$@"
