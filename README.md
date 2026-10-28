# EMNLP-shared-task-2026

This repository contains the code to reproduce the results of our submitted paper to EMNLP WMT2026 "Leveraging Verbalized Confidence in LLM-as-a-Judge for Segment-Level Translation Quality Score Prediction".

# Usage

`main.py` runs the pipeline that address subtask 2, from which we also derive which segment is error-free (subtask 3). For our contribution to subtask 1, run the `cot_tagged_span_annotation.py` using `cot_tagged_span_annotation.sh` (see our paper for further details).

Before executing `main.py`, fill in the following fields within the script:

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

**Remark:** We happen to use Qwen3.6-27B, Qwen3.8-27B-FP8, and Babel-
83B-chat, locally via vLLM; but feel free to make whatever adjustment that fits your pipeline.

# Training and testing

In `main.py`, the large commented-out section below `# training #` trains the regressions (`MLPRegressor`) on human annotations, while the section below `# testing #` loads and uses the trained models  (`model_full.predict(X)` for the model trained *with* confidence ratings, and `model_no_conf.predict(X_no_confidences)` for the one trained *without* them).

Citation
---------

Richard Servajean, Mohammad Golam Sohrab, Lucie Kunitomo-Jacquin, and Matiss Rikters. 2026. Leveraging Verbalized Confidence in LLM-as-a-Judge for Automated Translation Quality Evaluation. In Proceedings of the Eleventh Conference on Machine Translation, pages 1730–1738, Budapest, Hungary. Association for Computational Linguistics.

```bibtex
@inproceedings{servajean-etal-2026-leveraging,
    title = "Leveraging Verbalized Confidence in {LLM}-as-a-Judge for Automated Translation Quality Evaluation",
    author = "Servajean, Richard  and
      Sohrab, Mohammad Golam  and
      Kunitomo-Jacquin, Lucie  and
      Rikters, Matiss",
    editor = "Haddow, Barry  and
      Kocmi, Tom  and
      Koehn, Philipp  and
      Monz, Christof",
    booktitle = "Proceedings of the Eleventh Conference on Machine Translation",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.wmt-main.107/",
    pages = "1730--1738",
    ISBN = "979-8-89176-474-3",
    abstract = "This paper presents the AIST AIRC team{'}s contributions to subtasks 1 (segment-level error detection and span annotation), 2 (segment-level quality score prediction) and 3 (detection of error-free segments) of the automated translation quality evaluation systems shared task in WMT 2026. Our approach to address each of the 3 subtasks relies on an LLM-as-a-judge system. Specifically, for subtasks 2, we follow an LLM-as-a-judge approach within the Multidimensional Quality Metrics (MQM) error typology, with the novel addition of prompting the large language model (LLM) to verbalize its confidence in each error type assessment. We then train a Multilayer Perceptron (MLP) Regressor on error severities with and without associated confidence ratings to predict human annotations. First, our results provide further evidence for the relevance of integrating LLMs into automated translation quality evaluation pipelines. Second, this work aims to advance ongoing efforts to make these pipelines uncertainty-aware through the integration of verbalized confidence. Our code is freely available at https://github.com/sshrichard/WMT-Evaluation-Shared-Task-2026."
}
```
