GGUF=/Users/amaterasu/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct-GGUF/snapshots/b69aef112e9f895e6f98d7ae0949f72ff09aa401/Llama-3.2-1B-Instruct-Q4_K_M.gguf

../../../target/release/rayzor debug bench Main.hx  \
  --metric tok-per-s \
  -n 6 \
  --timeout 240 \
  --tier-start-interpreted true \
  --tier-promotion true \
  --tier-thresholds 1/20/5 \
  --tier-thresholds 1/20/2 \
  --tier-thresholds 1/30/5 \
  --cooldown-ms 15000 \
  -- "$GGUF" "Explain voronoi regions, and their connection to delauney computation and graph memory models. With coding examples. Describe vector graph database implementation" 5000 0.7