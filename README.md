# Explainable Individual Re-Identification

Modular deep learning codebase for wildlife individual re-identification (ReID), with:
- backbone finetuning (`train/finetune.py`)
- retrieval probing / benchmarking (`train/probe.py`)
- optional mask-based background removal
- optional Weights & Biases logging

## Overview

The repository supports two workflows:
- `finetune`: train a backbone with ArcFace loss on a train split and evaluate retrieval on a validation split.
- `probe`: benchmark retrieval methods (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `vismatch`) with pretrained or finetuned backbones.

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
│   │   ├── accumulation.py
│   │   └── checkpointing.py
│   ├── config_defaults.py
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

## Supported Models

Supported model identifiers:

- megadescriptor-t
- megadescriptor-l (the default)
- lynx_megadescriptorV3
- lynx_megadescriptorV4
- miewid
- dinov2
- dinov3

The legacy name megadescriptor is not accepted. Use an explicit supported
identifier in custom configurations.

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

Run Vismatch benchmark:

```bash
python train/probe.py --method vismatch
```

### Kaggle Jaguar Re-ID (new standalone pipeline)

This repository now includes a dedicated competition pipeline that keeps existing `train/probe` behavior unchanged:
- finetune backbone with ArcFace
- local validation (identity-balanced mAP on stratified train/val split)
- Stage A retrieval (`cosine` or `wildfusion`)
- optional Stage B Vismatch reranking
- strict Kaggle submission validation and CSV export
- optional PNG alpha-mask application (`alpha_mask.enabled`, default `true`)

Config:
- `config/kaggle_jaguar.yaml`

Alpha-mask mode (Kaggle-only):
- `alpha_mask.enabled: true` multiplies RGB with PNG alpha channel and caches masked RGB files.
- Applies to both finetuning and inference stages.
- Debug samples are saved under `<run_dir>/alpha_mask_debug/`.

Submission mode:
- `submission.mode: stage_a_plus_vismatch`: Stage A (`cosine` or `wildfusion`) + Stage B Vismatch reranking.
- `submission.mode: stage_a_only`: skip Vismatch entirely and submit only Stage A scores.
- Output files are mode-tagged, e.g. `submission_stage_a_only.csv` or `submission_stage_a_plus_vismatch.csv`.

Vismatch fusion controls (Kaggle pipeline):
- `submission.vismatch_fusion_mode`: `delta` (recommended), `blend`, or `replace`.
- `submission.vismatch_fusion_alpha`: fusion strength used by `delta`/`blend` (for Jaguar, start around `0.08-0.10` and validate).
- `submission.vismatch_min_stage_score`: only apply Vismatch fusion where Stage-A score is above threshold.
- `submission.vismatch_fusion_symmetrize`: enforce symmetric all-vs-all similarity matrix before submission.

Stage-A method:
- `stage_a.method: cosine` uses finetuned backbone embeddings + cosine matrix.
- `stage_a.method: wildfusion` runs calibrated WildFusion as Stage-A (`stage_a.wildfusion.*` settings).

Run:

```bash
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id
```

Useful flags:

```bash
# quick debug
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id --dry-run --pair-limit 2000

# faster full-ish iteration
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id --fast

# skip finetune and use existing checkpoint
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id --checkpoint /path/to/checkpoint.pth
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
- `benchmark`: method (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `vismatch`), metrics, cache
- WildFusion settings: `B` controls candidate pairs per query, `local_batch_size` controls pair-processing batches, and `local_top_k` controls ALIKED keypoints (default `512`).
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

Visualization option used by Vismatch keypoint match images:
- `visualization.vismatch_max_matches`

#### Vismatch Settings

`vismatch` runs a two-stage pipeline:
- Stage A (fast global retrieval) builds top-K candidates per query.
- Stage B reranks only those candidates with a configured local matcher.

The production path extracts each image once and matches cached features. The
optional pairwise Vismatch API is reserved for diagnostics because it would repeat
feature extraction for every candidate pair.

Config path:
- `benchmark.methods.vismatch`

Supported matcher profiles:
- `rdd-lightglue` (the migrated legacy RDD-LightGlue setup)
- `aliked-lightglue`
- `superpoint-lightglue`
- `loma` (Vismatch-managed LoMa-B)

Core options:
- `matcher`: selected Vismatch matcher profile
- `cache_dir`: matcher-specific per-image feature cache directory (`.npz`)
- `device`: `auto` | `cpu` | `cuda`
- `path_col`: metadata image path column
- `resize_max`, `top_k`, `matcher_threshold` (`null` selects the profile default: `0.01` for
  RDD/LightGlue and `0.10` for LoMa)
- `feature_matching_mode`: `feature_level` (production) or `pairwise` (diagnostics only)
- `batch_mode`: `batched` (production default) or `serial` (parity/debug reference)
- `match_batch_size`: candidate-pair batch size (default `16`)
- `extract_batch_size`: cached feature-extraction batch size (default `8`)
- `oom_backoff`: halve and retry the active CUDA batch on OOM (default `true`)
- Batched and serial matching show a pair-counted progress bar with throughput and ETA; OOM retries advance it only after successful completion.
- `stage_a_method`: `cosine` | `wildfusion` | `local_lightglue` | `linear_probe` | `efficient_probe`
- `candidate_k`: shortlist size from Stage A reranked by Vismatch

To run LoMa instead of RDD-LightGlue, keep `benchmark.method: "vismatch"` and set:

```yaml
benchmark:
  methods:
    vismatch:
      matcher: "loma"
      matcher_threshold: null
```

LoMa follows the Lynx reference protocol: Vismatch's LoMa-B model, right/bottom
padding to multiples of 14, normalized[-1,1] cached keypoints, mutual matching,
and confidence-sum normalization by the smaller keypoint count. Its weights are
managed and downloaded by Vismatch on first use.

Vismatch is pinned to commit
`4a743b75749a3770af59d275483ed341dea51ff0` in `requirements.txt`. Its matcher
weights are downloaded on first use. The wrapper is BSD-3-Clause, but wrapped
models may have separate licenses.
The old public `rdd` method and direct RDD repository settings are unsupported.
Use `vismatch` with `matcher: rdd-lightglue` when reproducing the migrated RDD
experiment. Feature caches are matcher/profile-specific and are regenerated when
the schema, matcher, preprocessing, keypoint budget, threshold, or checkpoint
identity changes.

Batched mode preserves the Stage-A shortlist and score protocol. Feature extraction
uses matcher-native spatial-shape buckets; pair matching groups candidate pairs across
queries by exact keypoint shape, so LoMa is not padded in a way that changes softmax
normalization. Incompatible
or singleton groups use the serial backend, and CUDA OOM retries halve the active batch
size. Timing metadata records configured/effective extraction and matching sizes.
Use `batch_mode: serial` for a direct parity reference before changing matcher profiles,
preprocessing, thresholds, or keypoint budgets.

## Training and Evaluation Outputs

### Finetune outputs

Under results/<run_id>/:

Canonical files:
- checkpoint-final.pth — model-only inference checkpoint
- checkpoint-final-full.pth — full resume checkpoint
- checkpoint-latest-full.pth — latest full resume checkpoint
- optional checkpoint-best.pth and checkpoint-best-full.pth
- periodic checkpoint-epoch-<n>.pth (controlled by output.save_every)
- safety_checks/ artifacts when enabled

Aggregate metrics CSV:
- results/train_metrics.csv
- Finetuning CSV rows for a completed run include `total_run_sec` and `total_run_min`.

For compatibility, finetuning also writes historical tagged forms such as
checkpoint-final_<dataset_tag>.pth. Probe and Jaguar checkpoint discovery
recognize both canonical and tagged model-only final checkpoints, while explicit
checkpoint paths always take precedence.

### Probe outputs

Under `benchmark_runs/<run_id>/`:
- `result.json`
- `config.snapshot.yaml`
- `safety_checks/` artifacts when enabled

Aggregate benchmark CSV:
- `benchmark_runs/benchmark_results.csv`

Optional visualizations:
- `visualizations/<run_id>/predictions_*.png`

## Checkpoints and Gradient Accumulation

The standard finetune-to-probe workflow uses checkpoint-final.pth. If automatic
discovery is enabled, the newest run is searched for the configured canonical
filename first and then for compatible tagged model-only checkpoints. Full
checkpoints are never selected for inference.

accumulation_steps controls optimizer updates in all three training loops:
finetune, linear_probe, and efficient_probe. The final partial group at the end
of an epoch is flushed so its gradients are not discarded.

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

## Known Constraints and Future Work

- Model weights are downloaded from Hugging Face on first use.
- Vismatch requires the pinned package, model-weight downloads, and usually CUDA for practical runtimes.
- Default configs contain environment-specific shared filesystem paths; update them
  for another machine.
- Masking and Vismatch matcher settings are dataset-dependent and should be validated rather than
  assumed to improve every dataset. Matcher ablations must keep Stage-A candidates, preprocessing,
  keypoint budgets, scoring, and evaluation metrics fixed.
- The test suite intentionally avoids CUDA, downloaded models, private datasets, and external
  Vismatch integration; optional environment-gated smoke and Lynx parity checks are required
  before changing the pinned Vismatch commit.
- Future experiment priorities are tracked in AGENTS.md.

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
