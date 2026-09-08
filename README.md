# EMNLP-shared-task-2026

This repository contains the necessary code to reproduce the results of our submitted paper to EMNLP WMT2026 "Leveraging Verbalized Confidence in LLM-as-a-Judge for Segment-Level Translation Quality Score Prediction".


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

In `main.py`, the large commented-out section below `# training #` trains the regressions (`MLPRegressor`) on human annotations, while the section below `# testing #` loads and uses the trained models  (`model_full.predict(X)` for the model trained *with* confidence ratings, and `model_no_conf.predict(X_no_confidences)` for the one trained *without* them).
