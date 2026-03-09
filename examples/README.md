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


### Langfuse setup
[quick start on langfuse](https://langfuse.com/docs/observability/get-started)

### run inside global uv environment
```shell
uv pip install ipykernel
uv run python -m ipykernel install --user --name sam3-uv --display-name "Python (sam3 uv)"
uv run jupyter lab --allow-root


uv sync --extra dev
uv run --extra dev python -m gdown -q "https://drive.google.com/drive/folders/1fxFUhKrNDHLTRQzAutzKPHFi51le-G8_?usp=drive_link" -O vids --folder
```