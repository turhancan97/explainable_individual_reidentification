# Explainable Individual Re-Identification

Modular deep learning codebase for wildlife individual re-identification (ReID), with:
- backbone finetuning (`train/finetune.py`)
- retrieval probing / benchmarking (`train/probe.py`)
- optional mask-based background removal
- optional Weights & Biases logging

## Overview

The repository supports two workflows:
- `finetune`: train a backbone with ArcFace loss on a train split and evaluate retrieval on a validation split.
- `probe`: benchmark retrieval methods (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `rdd`) with pretrained or finetuned backbones.

The code is organized into reusable modules under `reid/` and thin CLI entrypoints under `train/`.

## Repository Structure

```text
.
├── config/
│   ├── finetune_config.yaml
│   └── probe_config.yaml
├── models/
│   └── model.py
├── reid/
│   ├── data/
│   │   └── dataset_view.py
│   ├── engine/
│   │   ├── finetune_runner.py
│   │   └── probe_runner.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── features/
│   │   └── containers.py
│   ├── training/
│   │   └── checkpointing.py
│   └── utils/
│       ├── io.py
│       └── repro.py
├── tests/
│   └── test_shared_utils.py
├── train/
│   ├── finetune.py
│   └── probe.py
├── requirements.txt
└── environment.yml
```

## Installation

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate ex-reid
```

## Dataset Requirements

Configs assume a dataset root containing metadata CSV with split/label columns.

Default expected fields:
- `unique_name` (identity label)
- split column:
  - probe: `split-time_closed`
  - finetune: `split-time_closed`
- optional `mask` column for background removal (`dataset.no_background: true`)

Default path in configs:
- `/shared/sets/datasets/vision/czechlynx/CzechLynx_v2`

## Quick Start

### Finetune

```bash
python train/finetune.py
```

With explicit config:

```bash
python train/finetune.py --config config/finetune_config.yaml
```

### Probe / Benchmark

```bash
python train/probe.py
```

With explicit config:

```bash
python train/probe.py --config config/probe_config.yaml
```

Run linear probe:

```bash
python train/probe.py --method linear_probe
```

Run efficient probe:

```bash
python train/probe.py --method efficient_probe
```

Run RDD benchmark:

```bash
python train/probe.py --method rdd
```

## Configuration Guide

### `config/finetune_config.yaml`

Key blocks:
- `dataset`: root, metadata file, split values, `no_background`, `mask_col`
- `model`: backbone type
- `train`: epochs, batch size, AMP, deterministic mode, resume checkpoint
- `loss`: ArcFace parameters
- `scheduler`: cosine settings
- `output`: save frequency, best metric, CSV path
- `benchmark`: validation retrieval metrics (`top_k`, `mAP`)
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

### `config/probe_config.yaml`

Key blocks:
- `dataset`: root/splits + mask options
- `model`: type/mode/checkpoint behavior
- `benchmark`: method (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `rdd`), metrics, cache
- `visualization`: optional qualitative retrieval plots
- `output`: run folder + aggregate CSV
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

### Safety Checks

When `safety_checks.enabled: true`, both `finetune` and `probe` run pre-run validators before model loading:
- overlap check between split files (hard error)
- identity coverage report (seen/unseen identities)
- per-split class count histogram

Artifacts are saved under each run folder:
- `safety_checks/summary.json`
- `safety_checks/class_counts.csv`
- `safety_checks/class_count_histogram.png`

Classifier-based probe methods (`linear_probe`, `efficient_probe`) enforce closed-set identity coverage (query identities must exist in database identities).

#### Linear Probe Settings

`linear_probe` trains a softmax classifier on top of backbone embeddings and can optionally tune backbone weights.

Config path:
- `benchmark.methods.linear_probe`

Core options:
- `train_mode`: `all` | `partial` | `classifier`
- `epochs`, `batch_size`, `num_workers`, `accumulation_steps`
- `optimizer`: `sgd` | `adam` | `adamw`
- `lr`, `momentum`, `weight_decay`, `eta_min_scale`
- `eval_batch_size`, `eval_num_workers`
- `resume_checkpoint`
- `save_checkpoint` (default `false`), `save_every`, `final_checkpoint_name`
- `partial_rules`: per-model parameter-name patterns for partial unfreezing

Reported metrics for `linear_probe`:
- Retrieval: `top_k`, `mAP` (same benchmark path as other methods)
- Classification: `classification_top_1`, `classification_top_5`, `classification_top_10`

Example snippet:

```yaml
benchmark:
  method: "linear_probe"
  methods:
    linear_probe:
      train_mode: "classifier"   # all | partial | classifier
      epochs: 10
      optimizer: "sgd"
      lr: 0.001
      save_checkpoint: false
      partial_rules:
        default: ["layers.3", "norm"]
```

#### Efficient Probe Settings

`efficient_probe` applies a softmax head on top of ViT patch-token outputs:
- token source: `outputs.last_hidden_state[:, -number_of_patches:, :]`
- supports train modes: `all` | `partial` | `classifier`
- logs train/val loss and top-k metrics with tqdm progress bars
- when `visualization.enabled: true`, also saves a single attention-overlay grid from query images

Config path:
- `benchmark.methods.efficient_probe`

Core options:
- `train_mode`, `epochs`, `log_every`
- `batch_size`, `num_workers`, `accumulation_steps`
- `optimizer`, `lr`, `momentum`, `weight_decay`, `eta_min_scale`
- `dropout_rate`, `num_queries`, `d_out`
- `eval_batch_size`, `eval_num_workers`
- `resume_checkpoint`, `save_checkpoint`, `save_every`, `final_checkpoint_name`
- `partial_rules`

Visualization options used by efficient probe overlays:
- `visualization.attention_num_examples`
- `visualization.attention_average_queries`

Visualization option used by RDD keypoint match images:
- `visualization.rdd_max_matches`

#### RDD Settings

`rdd` runs a two-stage pipeline:
- Stage A (fast global retrieval) builds top-K candidates per query.
- Stage B reranks only those candidates with local RDD+LightGlue.

Config path:
- `benchmark.methods.rdd`

Core options:
- `repo_dir`: local path to your `rdd` repository
- `config_path`: RDD config file path
- `weights`: RDD model weights path
- `cache_dir`: per-image feature cache directory (`.npz`)
- `device`: `auto` | `cpu` | `cuda`
- `path_col`: metadata image path column
- `resize_max`, `top_k`
- `stage_a_method`: `cosine` | `wildfusion` | `local_lightglue` | `linear_probe` | `efficient_probe`
- `candidate_k`: shortlist size from Stage A reranked by RDD

## Training and Evaluation Outputs

### Finetune outputs

Under `results/<run_id>/`:
- `checkpoint-final.pth`
- `checkpoint-final-full.pth`
- `checkpoint-latest-full.pth`
- optional `checkpoint-best.pth` and `checkpoint-best-full.pth`
- periodic `checkpoint-epoch-<n>.pth` (controlled by `output.save_every`)
- `safety_checks/` artifacts when enabled

Aggregate metrics CSV:
- `results/train_metrics.csv`

### Probe outputs

Under `benchmark_runs/<run_id>/`:
- `result.json`
- `config.snapshot.yaml`
- `safety_checks/` artifacts when enabled

Aggregate benchmark CSV:
- `benchmark_runs/benchmark_results.csv`

Optional visualizations:
- `visualizations/<run_id>/predictions_*.png`

## Weights & Biases (W&B)

Both pipelines support optional W&B logging via config.

Enable:

```yaml
wandb:
  enabled: true
  project: "explainable-reid"
  entity: null
  group: null
  tags: []
  name: null
```

Logged data:
- finetune: train loss, validation metrics, learning rate
- probe: benchmark metrics/timings, metadata, optional visualization images
- linear_probe (within probe): per-epoch train loss, learning rate, classification + retrieval metrics
- efficient_probe (within probe): per-epoch train loss, learning rate, classification + retrieval metrics

## Reproducibility

- seed control available in both configs
- deterministic mode toggle (`deterministic: true/false`)
- finetune resume from full checkpoints via `train.resume_checkpoint`
- probe feature caching keyed by method/model/checkpoint/dataset signature and `no_background`

## Testing

Run unit tests:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Troubleshooting

- `ModuleNotFoundError: reid`
  - Run from repository root and use `python train/<script>.py`.
- mask decoding errors with `no_background: true`
  - Verify metadata has valid `mask` field (JSON string or COCO-RLE dict).
- CUDA mismatch or availability issues
  - Adjust device/AMP settings in config.
