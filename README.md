# EMNLP-shared-task-2026

WORK IN PROGRESS

# Usage

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

# Training and testing

In `main.py`, the large commented out part, below `# training #` corresponds to the learning the regression on human annotations, while the part below `# testing #` call the trained models, respectively here `model_full = joblib.load("model_full.joblib")` (the regression included confidence ratings) and here `predictions_test_full = model_full.predict(X)` (the regression did not include confidence ratings).
