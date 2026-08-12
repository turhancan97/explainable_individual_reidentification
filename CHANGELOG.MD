# Changelog

All notable repository changes are recorded here. The current development line is
kept under Unreleased until a project version is assigned.

## [Unreleased] - 2026-08-12

### Added

- Added a repository policy requiring AI agents to update AGENTS.md and
  CHANGELOG.MD after changes and important decisions.
- Added AGENTS.md with architecture, operating rules, configuration contracts,
  validation commands, compatibility conventions, and future work.
- Added lightweight supported-model constants and validation.
- Added regression tests for model defaults, checkpoint discovery, and gradient
  accumulation boundaries.

### Changed

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

- Prevented gradients from a final partial accumulation group from being discarded
  in finetuning, linear probing, and efficient probing.
- Ensured explicit checkpoint paths take precedence over automatic discovery.
- Ensured full resume checkpoints are not selected as model-only inference
  checkpoints.
- Replaced the stale megadescriptor shipped default with a supported identifier.

### Known Issues

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

### Fixed

- Removed direct RDD repository/config/weight path requirements from shipped
  configurations.
- Preserved matched-confidence score normalization and Jaguar Stage-A fusion
  semantics while changing the matcher backend.
- Preserved the Lynx LoMa scoring protocol: mutual-match confidences summed and
  divided by the smaller keypoint count, with the reference 0.10 filter threshold.
- Added a clear migration error for callers that still request the removed public
  `rdd` probe method.

### Known Issues

- The Vismatch backend depends on native feature/matcher components exposed by the
  pinned Vismatch commit; changing that dependency requires a parity check.
- LoMa requires its Vismatch-managed weights and the pinned LoMa dependencies;
  changing the LoMa wrapper or threshold requires a separate parity check.
- Private Lynx golden-subset parity and CUDA matcher smoke tests still require the
  external dataset, downloaded weights, and the ex-reid environment.
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
