# EMNLP-shared-task-2026

WORK IN PROGRESS

# usage

Before running `main.py`, fill in the following fields:

```python
client = AsyncOpenAI(
    base_url="base_url",   # API endpoint
    api_key="api_key",     # API key
)
```

Once the virtual environment is created (and activated), install the dependencies:
```bash
pip install -r requirements.txt
```

**Remark:** We happen to use Qwen (Qwen3.6-27B) locally via vLLM, feel free to make whatever adjustment that fits your pipeline.

