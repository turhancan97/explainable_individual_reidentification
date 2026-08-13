# Changelog

All notable repository changes are recorded here. The current development line is
kept under Unreleased until a project version is assigned.

## [Unreleased] - 2026-08-13

### Added

- Added self-contained `experiments/` artifacts for new probe and finetune runs, including manifests, metrics, timings, training history, and a central `reports/runs.csv` index.
- Added run-local visualization indexes, top-1/failure contact sheets, and `scripts/summarize_runs.py` for filtered Markdown/CSV reporting.
- Added Hydra 1.3 configuration entrypoints and dependency-light composition tests for probe and finetuning.
- Added configurable WildFusion ALIKED keypoint budgets via `local_top_k` (default 512), with forced keypoint counts and cache/experiment identity tracking.

- Added a repository policy requiring AI agents to update AGENTS.md and
  CHANGELOG.MD after changes and important decisions.
- Added AGENTS.md with architecture, operating rules, configuration contracts,
  validation commands, compatibility conventions, and future work.
- Added lightweight supported-model constants and validation.
- Added regression tests for model defaults, checkpoint discovery, and gradient
  accumulation boundaries.
- Kept configuration regression checks compatible with the user-preserved wildfusion Stage-A default.
- Added pair-counted Vismatch matching progress with ETA and retry-safe progress updates.
- Added dependency-light batching tests covering shape bucketing, candidate placement,
  partial batches, serial mode, score parity scaffolding, and CUDA OOM backoff.
- Added batched Vismatch extraction and feature-level matching for LoMa and
  RDD-LightGlue-compatible profiles.

### Changed

- Preserved historical generated artifacts and aggregate CSVs while routing future probe and finetune runs through the readable experiment layout.
- Updated automatic checkpoint discovery to search modern finetune experiments before legacy results, without changing explicit-path priority or model-only safety.
- Moved probe and finetune defaults to `conf/probe.yaml` and `conf/finetune.yaml`, making Hydra the primary single-run configuration interface.
- Preserved project-managed output directories and added fully resolved finetune configuration snapshots.
- Documented the required shared ex-reid conda environment path and activation
  instructions for future AI sessions.
- Standardized shipped finetune and probe defaults on megadescriptor-l.
- Added canonical checkpoint outputs while retaining historical tagged model-only
  checkpoint files.
- Updated probe and Jaguar checkpoint discovery to prefer canonical files and fall
  back to tagged final checkpoints.
- Updated README documentation for supported models, outputs, dependencies,
  checkpoint behavior, and accumulation behavior.

### Fixed

- Added running/completed/failed run lifecycle reporting so failed experiments retain a concise error manifest.
- Rejected unknown and misspelled Hydra override paths during configuration composition.
- Added `total_run_sec` and `total_run_min` to completed finetuning runs in the
  aggregate training CSV, while preserving incremental epoch writes.
- Fixed the missing `Sequence` typing import that prevented batched Vismatch
  probe runs from starting.
- Added an ex-reid-gated runtime smoke test that invokes batched extraction with a
  fake backend, catching runtime annotation/import failures before full benchmarks.
- Restored explicit batched Vismatch defaults in `conf/probe.yaml` while
  preserving the user-selected LoMa, Stage-A, and candidate settings.
- Prevented gradients from a final partial accumulation group from being discarded
  in finetuning, linear probing, and efficient probing.
- Ensured explicit checkpoint paths take precedence over automatic discovery.
- Ensured full resume checkpoints are not selected as model-only inference
  checkpoints.
- Replaced the stale megadescriptor shipped default with a supported identifier.

### Known Issues

- New visualization contact sheets require Pillow-readable image files and are omitted when no images are available.
- Historical `results/`, `benchmark_runs/`, and top-level visualization artifacts are intentionally not migrated.
- Probe and finetune no longer accept the legacy argparse flags or `--config`; existing scripts must use Hydra dotlist overrides.
- Hydra multirun sweeps are not supported by the project experiment contract.
- Full experiments require heavyweight dependencies, model downloads, and often
  CUDA-capable hardware.
- Vismatch downloads matcher-specific weights on first use and can be expensive for
  large datasets; wrapped model licenses must be checked individually.
- Default dataset paths are environment-specific; the shared ex-reid environment must
  contain an importable pinned Vismatch installation.
- Masking and RDD improvements are dataset-dependent and must be validated per
  animal/dataset.
- Integration coverage for real backbones, CUDA, Vismatch weights, and private Lynx
  parity is intentionally outside the lightweight unit-test suite.


### Added

- Added a pinned Vismatch matcher registry and feature-level backend for
  RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue.
- Added dependency-light tests for matcher validation, profile fingerprints, score
  normalization, empty matches, and Vismatch configuration defaults.
- Added canonical `FrameFeatures` shape/convention validation and an explicit
  migration error for the removed public `rdd` probe method.
- Kept the feature contract and profile registry dependency-light so base-environment
  tests do not import masking, CUDA, or downloaded matcher dependencies.
- Added Vismatch LoMa-B as the `loma` matcher profile, including its 14-pixel
  divisibility preprocessing, normalized-coordinate cache fields, mutual-match
  filtering, and profile-specific threshold default.

### Changed

- Migrated probe and Jaguar local matching from direct external RDD imports to
  configuration-driven Vismatch.
- Renamed the public probe method and Jaguar controls/artifacts from `rdd` to
  `vismatch`; `rdd-lightglue` remains the selected matcher identifier.
- Added feature-cache schema metadata and matcher-specific cache identity.
- Made feature-level matching the explicit production mode and retained pairwise
  matching only as a diagnostics API.
- Repaired the shared ex-reid Vismatch installation and recorded the pinned commit.
- Extended probe and Jaguar configuration examples so `matcher: loma` can be
  selected without changing the public `vismatch` method name.
- Made batched Vismatch execution the default with configurable extraction and
  matching batch sizes, cross-query exact shape bucketing, serial fallback, and CUDA
OOM backoff.
- Added effective/configured batch-size and batch-count timing fields without changing
  Stage-A candidate selection, candidate_k, matrix placement, or score normalization.

### Fixed

- Removed direct RDD repository/config/weight path requirements from shipped
  configurations.
- Preserved matched-confidence score normalization and Jaguar Stage-A fusion
  semantics while changing the matcher backend.
- Preserved the Lynx LoMa scoring protocol: mutual-match confidences summed and
  divided by the smaller keypoint count, with the reference 0.10 filter threshold.
- Added a clear migration error for callers that still request the removed public
  `rdd` probe method.
- Recovered from interrupted/corrupt feature-cache writes with atomic cache replacement
  and recomputation of invalid `.npz` entries.

### Known Issues

- The Vismatch backend depends on native feature/matcher components exposed by the
  pinned Vismatch commit; changing that dependency requires a parity check.
- LoMa requires its Vismatch-managed weights and the pinned LoMa dependencies;
  changing the LoMa wrapper or threshold requires a separate parity check.
- Private Lynx golden-subset parity and CUDA matcher smoke tests still require the
  external dataset, downloaded weights, and the ex-reid environment.
- Exact shape bucketing is deliberately conservative: heterogeneous feature shapes
  may fall back to serial processing, so throughput depends on the dataset and matcher.
- Batched-vs-serial CUDA parity and the 100% top-1 golden-subset gate remain environment-
  gated checks; lightweight tests do not prove numerical equivalence of downloaded models.
- Full available-split parity was measured on 2026-08-12 with 66 queries and 217
  gallery sequences: 98.48% top-1 ranking agreement, identical top-5 accuracy,
  mAP difference -0.0018, and mean absolute per-pair score difference 4.38e-05.
  One near-tie changed the top-1 result; strict 100% golden-subset parity is not
  yet established.

## Historical Features

- Modular ArcFace backbone finetuning with retrieval validation.
- Retrieval benchmarking using cosine similarity, WildFusion, local LightGlue,
  linear probes, efficient ViT probes, and RDD reranking.
- Optional COCO-RLE background removal and feature caching.
- Split-overlap and identity-coverage safety checks with saved reports.
- Optional Weights & Biases logging and qualitative retrieval, attention, and match
  visualizations.
- Standalone Kaggle Jaguar pipeline with stratified validation, alpha-mask support,
  Stage-A retrieval, optional RDD fusion, score validation, and submission export.
