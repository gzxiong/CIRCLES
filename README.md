# CIRCLES: Composed Image Retrieval for Causal Learning Example Selection

<p align="center">
  <strong>Official code for <em>Retrieving Counterfactuals Improves Visual In-Context Learning</em> (CVPR 2026)</strong>
</p>

<p align="center">
  CIRCLES improves visual in-context learning for vision-language models by combining standard correlational retrieval with attribute-guided composed retrieval for counterfactual-style example selection.
</p>

<p align="center">
  <a href="https://gzxiong.github.io/circles/">
    <img src="https://img.shields.io/badge/Homepage-coming_soon-0f766e" alt="Homepage">
  </a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg" alt="arXiv">
  </a>
</p>

<!-- When the arXiv page is available, replace the line above with a badge or icon, e.g.
<p align="center">
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg" alt="arXiv">
  </a>
</p>
-->

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Implemented Retrieval Methods](#implemented-retrieval-methods)
- [Batch Experiments](#batch-experiments)
- [Citation](#citation)

## Overview

CIRCLES improves visual ICL by combining two complementary retrieval signals:

- **Correlational retrieval**: standard image-neighbor retrieval.
- **Causal retrieval**: attribute-guided composed image retrieval that surfaces counterfactual-style examples.

This repository includes:

- a reusable `VisualICL` class for single-query inference,
- batch scripts for running experiments across supported datasets,
- optional CLIP embedding precomputation utilities.


## Environment Setup

```bash
# 1) Install vLLM first by following the official guide:
#    https://docs.vllm.ai/en/stable/getting_started/installation/index.html

# 2) Install the remaining project dependencies
pip install -r requirements.txt
```

## Data Preparation

`src/load_data.py` resolves dataset folders relative to the repository root and automatically downloads missing supported datasets on first use.

Expected default locations:

- `data/cub`
- `data/flowers`
- `data/okvqa`
- `data/vizwiz`

Supported datasets: `cub`, `flowers`, `okvqa`, `vizwiz`.

If you want to precompute CLIP embeddings ahead of time:

```bash
python src/precompute_embeddings.py --data cub
```

## Quick Start

### 1. Build a VLM and `VisualICL`

```python
from vllm import LLM, SamplingParams
from src.visual_icl import VisualICL

vlm = LLM(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 36, "video": 0},
)

icl = VisualICL(
    vlm=vlm,
    method="circles",  # one of: none, random, rices, muier, mmices, circles
    sampling_params=SamplingParams(temperature=0.0, max_tokens=64),
    default_k=16,
    num_attributes=1,
    attribute_k=16,
)
```

### 2. Run inference on a new query

```python
from src.load_data import load_dataset

train_ds = load_dataset("cub", split="train")
test_ds = load_dataset("cub", split="test")
item = test_ds[990]

result = icl.predict(
    question=item["question"],
    query_image=item["imgpath"],
    train_dataset=train_ds,
    task=test_ds.task,  # "vqa" or "cls"
    options=getattr(test_ds, "options", None),
)

print(result["answer"])
print(result["retrieved_examples"])
```

### 3. Switch retrieval methods with the same API

```python
for method in ["none", "random", "rices", "muier", "mmices"]:
    icl.method = method
    result = icl.predict(
        question=item["question"],
        query_image=item["imgpath"],
        train_dataset=train_ds,
        task=test_ds.task,
        options=getattr(test_ds, "options", None),
        k=32,
    )
    print(method, result["answer"])
```

### 4. Optionally provide attributes or retrieved examples explicitly

For `circles`, you can pass attributes directly to skip automatic attribute identification:

```python
icl.method = "circles"
result = icl.predict(
    question=item["question"],
    query_image=item["imgpath"],
    train_dataset=train_ds,
    task=test_ds.task,
    options=getattr(test_ds, "options", None),
    attributes=["Small size"],
)
print(result["answer"])
```

For retrieval-based methods, you can pass `retrieved_examples` to bypass internal retrieval.

Expected format:

- `none`, `random`, `rices`, `muier`, `mmices`: a list of training items.
- `circles`: a dictionary with keys `original_retrievals` and `composed_retrievals`. Each item in `composed_retrievals` contains `attribute`, `modified_caption`, and `retrieved_items`.

## Implemented Retrieval Methods

- **none**: zero-shot prompting
- **random**: random in-context examples
- **rices**: image-image nearest-neighbor retrieval
- **muier**: joint image-image and image-text retrieval
- **mmices**: image-based candidate retrieval with question-guided reranking
- **circles**: standard image retrieval plus attribute-guided composed retrieval

## Batch Experiments

Run batch inference:

```bash
python src/run_batch_inference.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --data cub \
  --method circles \
  --k 16 \
  --num_attributes 1 \
  --attribute_k 16
```

Output path:
`results/<dataset>/<model_name>/<method>_*.jsonl`

Evaluate predictions:

```bash
python src/evaluate_results.py \
  --data cub \
  --pred_file results/cub/Qwen_Qwen2.5-VL-3B-Instruct/circles_16_1_16.jsonl
```

For `vizwiz`, `src/evaluate_results.py` uses normalization and scoring conventions aligned with the official VizWiz VQA evaluation code to keep results comparable.

## Citation

```bibtex
@inproceedings{xiong2026retrieving,
  title     = {Retrieving Counterfactuals Improves Visual In-Context Learning},
  author    = {Guangzhi Xiong and Sanchit Sinha and Zhenghao He and Aidong Zhang},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026}
}
```
