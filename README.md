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
├── conf/
│   ├── finetune.yaml
│   └── probe.yaml
├── config/
│   └── kaggle_jaguar.yaml
├── experiments/
│   ├── probe/
│   └── finetune/
├── reports/
│   └── runs.csv
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

Override configuration values with Hydra dotlist syntax:

```bash
python train/finetune.py train.epochs=10 train.batch_size=32
```

### Probe / Benchmark

```bash
python train/probe.py
```

Select methods and override nested settings with Hydra:

```bash
python train/probe.py benchmark.method=linear_probe
python train/probe.py benchmark.method=efficient_probe
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
```

For the reproducible Slurm ablation grid, use the separate launcher:

```bash
# Submit 54 tasks with the default maximum of four concurrent jobs.
bash probe-parallel.sh

# Inspect the complete task table without submitting jobs.
bash probe-parallel.sh --list-tasks

# Print the array submission command without submitting it.
PROBE_PARALLEL_DRY_RUN=1 bash probe-parallel.sh
```

`probe-parallel.sh` leaves `probe.sh` unchanged and evaluates cosine, WildFusion,
local LightGlue, linear probe, efficient probe, and default/fine-tuned LoMa and
RDD-LightGlue across `candidate_k` values `10, 50, 100, 250, 500, 1000`.
The concurrency cap is `MAX_CONCURRENT_JOBS=4` near the top of the file. The two
custom Vismatch checkpoint paths are also editable there; custom variants use
`checkpoint_components=matcher_only`, while default variants use Vismatch-managed
weights. The launcher fails before submission if either custom checkpoint path is
missing. Use `MAX_CONCURRENT_JOBS=2 bash probe-parallel.sh` to change the throttle.

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

Hydra is the primary configuration interface for probe and finetuning. The shipped
defaults are in `conf/probe.yaml` and `conf/finetune.yaml`; Hydra resolves their
interpolations before the runner starts. Nested overrides use `key=value` dotlist
syntax and values are type-converted by OmegaConf.

Unknown keys and misspelled paths fail immediately. The old `--config`, `--method`,
`--dataset-root`, and related argparse flags are no longer supported for these two
entrypoints. The Jaguar submission script intentionally keeps its separate
argparse plus `config/kaggle_jaguar.yaml` workflow.

Hydra is configured not to change the working directory or replace project-managed
artifact paths. Probe and finetune continue writing their normal run directories and
store a fully resolved `config.snapshot.yaml` in each run. Hydra multirun sweeps are
not part of the supported experiment workflow.

### `conf/finetune.yaml`

Key blocks:
- `dataset`: root, metadata file, split values, `no_background`, `mask_col`, and `image_variant` (`background` or `no_background`)
- `model`: backbone type
- `train`: epochs, batch size, AMP, deterministic mode, resume checkpoint
- `loss`: ArcFace parameters
- `scheduler`: cosine settings
- `output`: experiment root, save frequency, best metric, and legacy aggregate CSV path
- `reporting`: run-index path and reporting enablement
- `benchmark`: validation retrieval metrics (`top_k`, `mAP`)
- `safety_checks`: pre-run split validation (`enabled`)
- `wandb`: optional experiment logging

### `conf/probe.yaml`

Key blocks:
- `dataset`: root/splits + mask options and explicit `image_variant` (`background` or `no_background`)
- `model`: type/mode/checkpoint behavior
- `benchmark`: method (`cosine`, `wildfusion`, `local_lightglue`, `linear_probe`, `efficient_probe`, `vismatch`), metrics, cache
- `benchmark.candidate_k`: single comparison budget (default `100`) used for Vismatch
  candidates, WildFusion refinement, and the `mAP_at_k`, `rerank_mAP_at_k`, and
  `recall_at_k` evaluation cutoff.
- WildFusion settings: `local_batch_size` controls pair-processing batches and
  `local_top_k` controls ALIKED keypoints (default `512`). Its refinement `B` is derived
  from `benchmark.candidate_k`.
- Local LightGlue also receives its refinement budget from `benchmark.candidate_k`; its
  old method-specific `B` override is no longer supported.
- `visualization`: optional qualitative retrieval plots
- `output`: experiment root, legacy run folder, and aggregate CSV
- `reporting`: central run-index path
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
  RDD/LightGlue and `0.10` for LoMa). For Vismatch, `resize_max` is the target long-side
  resolution; the shipped parity default is `512`.
- Vismatch preprocessing converts images to RGB float32 tensors in `[0, 1]` and resizes
  directly with bilinear `F.interpolate`. RDD-LightGlue, ALIKED-LightGlue, and
  SuperPoint-LightGlue floor both dimensions to multiples of 32. LoMa floors both
  dimensions to multiples of 14 because its DINOv2-L/14 descriptor requires patch
  divisibility. Source image dimensions are retained separately for provenance and
  visualization. Existing caches from the previous generic `/32` LoMa path are
  incompatible and will not be reused.
- Cosine, WildFusion, local LightGlue, linear probe, and efficient probe retain their
  existing square-resize protocols.
- `feature_matching_mode`: `feature_level` (production) or `pairwise` (diagnostics only)
- `batch_mode`: `batched` (production default) or `serial` (parity/debug reference)
- `match_batch_size`: candidate-pair batch size (default `16`)
- `extract_batch_size`: cached feature-extraction batch size (default `8`)
- `oom_backoff`: halve and retry the active CUDA batch on OOM (default `true`)
- Batched and serial matching show a pair-counted progress bar with throughput and ETA; OOM retries advance it only after successful completion.
- `stage_a_method`: `cosine` | `wildfusion` | `local_lightglue` | `linear_probe` | `efficient_probe`
- `candidate_k`: shared benchmark budget for Vismatch shortlists, WildFusion `B`, Local LightGlue `B`, and evaluation cutoffs

Vismatch probe scoring is shortlist-constrained, matching the WildFusion baseline:
only the `candidate_k` pairs are scored by Vismatch; all unscored matrix positions
are `-inf`. The primary metrics and visualizations use this same matrix. The run
records `score_matrix_policy=shortlist_only_neg_inf`, `num_candidate_pairs`,
`num_unscored_pairs`, and `candidate_fraction`. WildFusion derives its refinement
budget from the same `benchmark.candidate_k`, so comparison runs cannot accidentally
use different candidate/refinement budgets. These are not full-gallery matcher
metrics, so always interpret them together with candidate recall.

Checkpoint selection options:
- `checkpoint_source`: `default` (bundled Vismatch weights) or `custom`.
- `checkpoint_path`: an exact `.safetensors`, `.pth`, `.pt`, or epoch directory; no newest-epoch auto-selection is performed.
- `checkpoint_components`: `auto`, `matcher_only`, `extractor_only`, or `full`.
- `loma_arch`: explicit LoMa variant, default `LoMa-B`.

For the current LightGlue-only RDD checkpoint, use:

```bash
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=rdd-lightglue benchmark.methods.vismatch.checkpoint_source=custom benchmark.methods.vismatch.checkpoint_path=/path/to/epoch_15 benchmark.methods.vismatch.checkpoint_components=matcher_only
```

The resolver detects components from tensor schemas and optionally validates `checkpoint_manifest.json`; it does not trust filenames such as `model.safetensors` or `model_1.safetensors`. RDD-LightGlue may combine custom RDD and LightGlue files, while a LightGlue/RDD checkpoint is never accepted for LoMa. Custom component hashes are included in feature-cache identities and run manifests.

To run LoMa instead of RDD-LightGlue, keep `benchmark.method: "vismatch"` and set:

```yaml
benchmark:
  methods:
    vismatch:
      matcher: "loma"
      matcher_threshold: null
```

LoMa follows the Lynx reference protocol: Vismatch's LoMa-B model, target long-side
resize with dimensions floored to multiples of 14, normalized[-1,1] cached keypoints,
mutual matching, and confidence-sum normalization by the smaller keypoint count. Its
weights are managed and downloaded by Vismatch on first use.

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

### Experiment artifacts and reporting

New probe and finetune runs are stored under `experiments/` using dataset, split,
model, method, matcher, timestamp, and configuration-hash components. For example:

```text
experiments/probe/CzechLynx_v2/CzechLynx/split-time_closed/
  megadescriptor-l/vismatch/loma/20260813T142530Z_a1b2c3d4/
    config.snapshot.yaml
    run_manifest.json
    metrics.json
    timings.json
    visualizations/index.csv
    visualizations/contact_sheet_top1.png
```

Finetune run directories also contain canonical checkpoints and
`training_metrics.csv`. Each manifest records status, resolved configuration, git
commit, environment information, dataset sizes, metrics, timings, and artifact paths.
Failed runs are retained with status and error information.

Existing aggregate files remain active for compatibility:

- `results/.../train_metrics.csv`
- `benchmark_runs/benchmark_results.csv`
- `reports/runs.csv` (one row per modern run)

New visualizations are stored with their run under `visualizations/`. The local
`index.csv` connects query/database indices, identities, ranks, scores, correctness,
and image paths. Top-1 and failure contact sheets are generated when images are
available. Historical `visualizations/<run_id>/` folders are not migrated.

Summarize runs with:

```bash
python scripts/summarize_runs.py --dataset CzechLynx_v2
python scripts/summarize_runs.py --method vismatch --matcher loma
python scripts/summarize_runs.py --sort-by top_1 --format markdown
```

For compatibility, tagged model-only checkpoints remain readable. Automatic probe
discovery searches the new `experiments/finetune/` root first and then historical
`results/`; explicit checkpoint paths always take precedence.

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
- Cache identities also include the resolved dataset root, metadata file, and explicit `dataset.image_variant` (`background` or `no_background`) so normal and pre-masked features cannot be reused interchangeably.

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

## Research-validity reporting

The reported retrieval metrics now follow a documented primary/diagnostic split:

- `mAP` includes every query. Queries without a relevant gallery identity contribute
  AP=0; `mAP_eligible` retains the eligible-query-only diagnostic, while
  `mAP_query_coverage`, `num_queries_with_gallery_match`, and
  `num_queries_without_gallery_match` expose coverage.
- `mAP` and `mAP_eligible` are reported only when `score_coverage` is `1.0`. A
  shortlist method scores `candidate_k` of the gallery and leaves the rest at `-inf`,
  ordered by original database index; grading that tail measures metadata row order
  rather than the matcher, so both fields become `nan` there. Use `mAP_at_k`.
- `mAP_at_k` is the primary retrieval metric for shortlist methods and is computed the
  same way for full-matrix methods, keeping `cosine`, `wildfusion`, and `vismatch`
  comparable. It truncates at `benchmark.candidate_k`, gives no credit to unscored
  positions, and divides by `min(relevant, k)` so a query whose identity never reached
  the shortlist scores 0.
- The retrieval result splits into three readable parts: `recall_at_k` (did the
  shortlist contain the identity at all), `rerank_mAP_at_k` (given that it did, how well
  was it ordered), and `mAP_at_k` (end-to-end). Matcher ablations should compare
  `rerank_mAP_at_k`, which does not charge every matcher for the same Stage-A misses.
- Cutoffs are validated before model loading: Vismatch `top_k` values must fit inside
  `benchmark.candidate_k`. The old independent `benchmark.map_at_k`, Vismatch
  `candidate_k`, and WildFusion `B` overrides are unsupported; use `benchmark.candidate_k`.
- Each probe run writes `scores.npz`, a sparse COO record of the scored matrix entries,
  so metrics can be recomputed without repeating a matcher run.
- Historical `mAP` values in `reports/runs.csv` predate this gate, are not comparable
  across methods, and cannot be recomputed because those runs did not persist scores.
- Linear and efficient probes report identity-level retrieval as primary. Their
  image-level metrics remain available as `image_top_1`, `image_top_5`, `image_top_10`,
  and `image_mAP` diagnostics. Identity scores are read directly from the classifier's
  per-identity output columns, so a gallery holding many images per identity no longer
  fails metric computation. Both probes stay closed-set and are not directly comparable
  to the retrieval methods.
- All ranking and visualization paths use deterministic descending score order with
  original database index as the tie-breaker, including the run-local
  `visualizations/index.csv`, so the index resolves ties identically to the prediction
  grid it annotates and to the reported metrics.
- Vismatch and WildFusion use shortlist-constrained ranking. Vismatch scores only
  Stage-A candidates; unscored positions are `-inf` and are excluded from the final
  ranking. Vismatch reports `candidate_hit_rate`/`candidate_recall_at_k` plus
  `num_candidate_pairs`, `num_unscored_pairs`, and `candidate_fraction`.

Split safety checks retain path-overlap detection and now compute SHA-256 hashes for
resolved image files. Identical content across protected splits fails closed, with
sample paths and missing/unreadable files recorded in `safety_checks/summary.json`.
Feature caches similarly include image content, metadata, preprocessing, image variant,
model/checkpoint weights, and matcher-profile identities. This adds I/O but prevents
stale features when a file or model changes at the same path.

Automatic checkpoint discovery recursively searches `experiments/finetune/` before
legacy `results/`, ignores failed/incomplete runs and full resume checkpoints, and
prefers completed canonical model-only files before tagged historical files. Explicit
checkpoint paths retain highest priority. Finetune reports reload the best checkpoint
for primary metrics and retain final-epoch metrics separately. The current test split
is still the model-selection split; this limitation has not been changed.

  - Adjust device/AMP settings in config.
