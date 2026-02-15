### vllm serving setup
```bash
uv run vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 1 \
  --allowed-local-media-path / \
  --enforce-eager \
  --port 8002 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 32768
```