# Explainable Individual Re-Identification

Modular deep learning codebase for wildlife individual re-identification (ReID), with:
- backbone finetuning (`train/finetune.py`)
- retrieval probing / benchmarking (`train/probe.py`)
- optional mask-based background removal
- optional Weights & Biases logging

## Overview

The repository supports two workflows:
- `finetune`: train a backbone with ArcFace loss on a train split and evaluate retrieval on a validation split.
- `probe`: benchmark retrieval methods (global cosine, WildFusion, local LightGlue) with pretrained or finetuned backbones.

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
- `wandb`: optional experiment logging

### `config/probe_config.yaml`

Key blocks:
- `dataset`: root/splits + mask options
- `model`: type/mode/checkpoint behavior
- `benchmark`: method (`cosine`, `wildfusion`, `local_lightglue`), metrics, cache
- `visualization`: optional qualitative retrieval plots
- `output`: run folder + aggregate CSV
- `wandb`: optional experiment logging

## Training and Evaluation Outputs

### Finetune outputs

Under `results/<run_id>/`:
- `checkpoint-final.pth`
- `checkpoint-final-full.pth`
- `checkpoint-latest-full.pth`
- optional `checkpoint-best.pth` and `checkpoint-best-full.pth`
- periodic `checkpoint-epoch-<n>.pth` (controlled by `output.save_every`)

Aggregate metrics CSV:
- `results/train_metrics.csv`

### Probe outputs

Under `benchmark_runs/<run_id>/`:
- `result.json`
- `config.snapshot.yaml`

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

