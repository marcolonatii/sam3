### vllm serving setup for `sam3_agent.ipynb`
```bash
uv run vllm serve Qwen/Qwen3-VL-8B-Thinking \
  --tensor-parallel-size 1 \
  --allowed-local-media-path / \
  --enforce-eager \
  --port 8002 \
  --gpu-memory-utilization 0.7 \
  --max-model-len 32768
```

### ReAct_agent
- an react agent designed to do video analytics.
- tooling: get_frame, tracking, positional understanding
- features: langfuse observation