# Changelog

All notable repository changes are recorded here. The current development line is
kept under Unreleased until a project version is assigned.

## [Unreleased] - 2026-08-13

### Added

- Added shortlist-safe retrieval metrics `mAP_at_k`, `rerank_mAP_at_k`, `recall_at_k`,
  `num_queries_with_relevant_in_top_k`, and `score_coverage`, computed identically for
  full-matrix and shortlist-constrained methods so matchers stay comparable.
- Added `benchmark.map_at_k` (default `100`) as the shared evaluation cutoff, and a
  startup validator that rejects any `top_k`/`map_at_k` larger than the Vismatch
  `candidate_k` shortlist.
- Added sparse score-matrix persistence (`scores.npz`, COO of finite entries) plus
  `save_score_matrix`/`load_score_matrix`, so run metrics can be recomputed without
  repeating a matcher run.
- Added `wildfusion_B` and `wildfusion_refined_fraction` so the WildFusion refinement
  budget is auditable next to Vismatch's `candidate_k`.
- Added regression coverage for full-matrix mAP suppression, unscored-tail credit,
  shortlist-reach versus ordering separation, and score-matrix round-tripping.
- Added self-contained `experiments/` artifacts for new probe and finetune runs, including manifests, metrics, timings, training history, and a central `reports/runs.csv` index.
- Added run-local visualization indexes, top-1/failure contact sheets, and `scripts/summarize_runs.py` for filtered Markdown/CSV reporting.
- Added Hydra 1.3 configuration entrypoints and dependency-light composition tests for probe and finetuning.
- Added configurable WildFusion ALIKED keypoint budgets via `local_top_k` (default 512), with forced keypoint counts and cache/experiment identity tracking.
- Added explicit `dataset.image_variant` provenance to standard and Vismatch feature-cache identities, preventing normal and pre-masked feature reuse; added regression coverage.

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
- Added research-validity regression coverage for stable ranking, unmatched-query
  mAP, identity-level probes, content-aware caches, duplicate-content safety,
  recursive checkpoint discovery, accumulation scaling, and calibration policy.
- Added SHA-256 fingerprint helpers and explicit WildFusion calibration diagnostics.
- Added dependency-light Vismatch checkpoint discovery for exact files and epoch directories, including RDD extractor, LightGlue, and LoMa component schemas.
- Added an optional ex-reid-gated smoke test for constructing RDD-LightGlue with a custom checkpoint and aligned Hydra regression coverage with the preserved LoMa default.
- Added `probe-parallel.sh`, a self-submitting capped Slurm array for 54 explicit
  method/checkpoint/`candidate_k` comparisons. It preserves `probe.sh`, validates
  custom LoMa and RDD-LightGlue checkpoint paths, supports task listing and dry-run
  inspection, and records the resolved task command in each array log.
- Added a dedicated `logs/parallel_run/` destination for parallel-array stdout and
  stderr so those logs are separate from ordinary probe logs.

### Changed

- Changed Vismatch extraction to preprocess each image once. Batch bucketing called
  `prepared_shape()`, which ran the full resize only to read the output shape, and
  extraction then repeated it, costing a measured 8.6 minutes per 39,760-image run.
  `prepare_image()` now returns a `PreparedImage` carrying the tensor and both sizes;
  pending buckets hold that resized tensor instead of the full-resolution source, cutting
  a sampled image from 18.0 MB to 3.2 MB. `prepared_shape` was removed, `extract_frame`
  and `extract_frames_batch` remain as wrappers, and `extract_prepared` now runs under
  `torch.no_grad()` to match the batched path. Features are bit-identical: verified
  against the pre-refactor call sequence for single, same-shape batched, and mixed-shape
  fallback extraction, so existing feature caches remain valid.
- Changed probe and finetune runs to hash each file once instead of three times. Split
  safety checks, `dataset_digest`, and the Vismatch `_cache_key` each read every image, so
  a 39,760-image probe run performed three full SHA-256 passes. `file_digest_cache()` is an
  explicit context manager wrapping a run, keyed on device/inode/size/mtime so the three
  call sites share entries despite building paths differently. Digests are unchanged and
  all cache identities are byte-identical; measured 20x on repeated hashing, saving roughly
  7 minutes per run warm and around 16 minutes cold.
- Decided against memoizing `sha256_file` process-wide: tmpfs reuses a single `st_mtime_ns`
  across rapid same-size rewrites, so an implicit global cache could return a stale digest
  and defeat the content-addressing these hashes exist to provide. Caching is therefore
  opt-in and run-scoped, and unscoped calls re-read the file exactly as before.
- Changed `mAP` and `mAP_eligible` to report `nan` unless every score-matrix position
  carries a real score. On a shortlist matrix the `-inf` tail is ~99.6% of the row and is
  ordered by original database index, so the previous value largely measured metadata row
  adjacency: across Vismatch runs whose `top_1` spanned 0.360-0.412, `mAP` stayed inside
  0.0463-0.0469, versus 0.0168 for a random ranking. Use `mAP_at_k` for those methods.
- Decided that all retrieval metrics are truncated at a single shared cutoff rather than
  ranking the full gallery. `top_k` already behaved this way, which is why it stayed
  trustworthy while `mAP` did not; `mAP_at_k` makes the rest of the metric set consistent.
  `mAP_at_k` divides by `min(relevant, k)` and remains end-to-end, while `rerank_mAP_at_k`
  divides by the hits inside the scored top-k to isolate ordering from shortlist reach.
- Consolidated the Hydra probe comparison budget under `benchmark.candidate_k`
  (default `100`). Vismatch shortlist selection, WildFusion refinement `B`, Local
  LightGlue refinement `B`, and
  `mAP_at_k`/`rerank_mAP_at_k`/`recall_at_k` now derive from this single value.
  The old independent `benchmark.map_at_k`, Vismatch `candidate_k`, and WildFusion
- Added dependency-light regression coverage for shared-budget resolution, database-size
  capping, invalid values, removed overrides, and preserved metric/provenance fields.
- Recorded that historical `mAP` values in `reports/runs.csv` and existing run manifests
  predate this gate and are not comparable across methods. They are not rewritten, and
  they cannot be recomputed because those runs did not persist their score matrices.
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
- Documented that primary mAP includes every query, with `mAP_eligible` and
  coverage fields retained for interpretation; linear and efficient probes now
  make identity-level metrics primary and retain image-level diagnostics.
- Documented deterministic original-index tie-breaking, candidate recall, and
  content-hash cache/split identities.
- Migrated RDD-family Vismatch matcher profiles to the Lynx-compatible float-tensor
  preprocessing protocol with a 512-pixel target long side and `/32` spatial rounding.
- Kept LoMa on its matcher-specific Lynx fine-tuning protocol: the same float-tensor
  interpolation, but `/14` spatial rounding for its DINOv2-L/14 descriptor.
- Preserved the existing square-resize protocols for cosine, WildFusion, local
  LightGlue, linear probe, and efficient probe.

### Fixed
- Fixed parallel Slurm array tasks failing with `Permission denied` when they used
  the copied spool script's directory. The launcher now prefers `SLURM_SUBMIT_DIR`
  before creating `logs/parallel_run/`.
- Fixed Vismatch discarding its Stage-A method metrics. `run_vismatch_benchmark` returns a
  fresh metrics dict, so Stage-A results are now merged after it under a
  `stage_a_<method>_` prefix, mirroring the existing Stage-A timing keys. Values are not
  coerced to float because some metrics are string diagnostics. This keeps Stage-A
  shortlist provenance such as `wildfusion_B` on the run instead of dropping it.
- Fixed the unreachable `warn_only_unseen` branch in split safety checks. It was only ever
  passed as `not require_closed_set`, so it was `True` exactly when the guarded block was
  skipped. Unseen identities are now always reported, with `require_b_labels_in_a` choosing
  the response: closed-set classification fails, open-set retrieval warns and continues.
  The dead parameter was removed from the signature and both call sites.
- Fixed a `NameError` when a finetune resumed from a checkpoint with no epochs left to run.
  `range(start_epoch, train.epochs)` was empty, so the post-loop code read the unbound epoch
  variable `metrics`. `validate_resume_epochs` now rejects that configuration up front with
  an actionable message, before any training setup or checkpoint writing, and the epoch
  metrics variable is pre-initialized alongside its neighbours. Decided to fail rather than
  train zero epochs and still emit final checkpoints and a completed manifest, which would
  hide an unraised `train.epochs`.
- Fixed run-local `visualizations/index.csv` ranking with `argsort()[::-1]`, which reverses
  a stable ascending sort and therefore resolved ties in descending database-index order,
  the opposite of the documented policy used by the prediction grid and the metrics. The
  index now uses `stable_rank_1d`. Shipped runs used `visualization.top_k: 3` with distinct
  top-3 scores, so existing artifacts are unaffected and are not regenerated; the defect
  would have surfaced on tied scores, such as a matcher returning 0.0 when no confidence
  clears its threshold, or a cutoff reaching past the scored shortlist.
- Fixed identity-level probe metrics indexing the classifier's per-identity output axis
  with a per-database-image boolean mask. `linear_probe` and `efficient_probe` raised
  `IndexError` at the end of every epoch whenever the gallery held more images than
  identities, so neither method could complete a run; `reports/runs.csv` contains no probe
  rows as a result. Identity columns are now selected directly, which is what the previous
  per-image maximum was reaching for: a classifier emits one probability per identity, so
  every image of that identity carried the same value and the maximum was a no-op. Added a
  guard rejecting database label indices outside the classifier head, and regression
  coverage for the image-count/class-count mismatch that triggered the crash.
- Fixed two Vismatch preprocessing test stubs that bypass `__init__` and had not been given
  the new `preprocessing_divisor` attribute, and corrected their `FakeExtractor` to return
  one dict per batch item as native `RDD.extract` does rather than a dict of stacked
  tensors. The serial/batched parity stub now derives values from image content so an
  item/feature misalignment fails the test instead of being hidden by identical
  placeholders. Production Vismatch code was correct and is unchanged.
- Fixed probe final metric printing so string diagnostic fields such as
  `vismatch_cache_fingerprint` no longer fail a completed run during finalization.
- Added regression coverage for mixed numeric and string metric reporting.
- Fixed a probe feature-cache startup regression where the cache key was referenced
  before construction.
- Added regression coverage that verifies cache keys include model-weight identity.
- Fixed probe startup calibration using an undefined dataset variable; the original
  dataset is now passed to calibration split selection.
- Fixed failed-run index creation when the `reports/` parent directory is absent, so
  reporting errors cannot mask the original probe exception.

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
- Made best-checkpoint metrics primary in finetune reports while retaining
  final-epoch metrics and checkpoint provenance.
- Removed finite Stage-A fallback scores and the `-1e9` sentinel for non-candidate
  Vismatch positions, using shortlist-only `-inf` scores and candidate-hit/recall
  reporting instead.
- Corrected accumulation loss scaling for full and partial groups in all three
  training loops.
- Added duplicate-image-content split failures and recursive completed-run
  checkpoint discovery with model-only precedence.
- Added Hydra-selectable custom Vismatch checkpoint loading with strict component validation, default-component fallback, content-hash cache isolation, and run-manifest provenance.
- Changed Vismatch probe ranking to initialize non-candidate positions to `-inf`, matching
  WildFusion's shortlist-constrained protocol; added score-policy and pair-coverage
  reporting for fair baseline comparisons.
- Added regression coverage for invalid matcher scores, full/partial shortlist matrices,
  deterministic `-inf` ties, and content-hash-compatible batching fixtures.
- Invalidated the previous Vismatch PIL-resize and LoMa `/14` feature semantics by
  bumping the feature schema and including `lynx_finetuning_v1` in matcher profiles
  and cache fingerprints.
- Corrected Vismatch feature metadata so keypoints and LightGlue image sizes use
  processed `/32` coordinates while source dimensions remain available for provenance.
- Corrected LoMa feature extraction to use processed `/14` coordinates and matcher-specific
  cache/profile identity, preventing the LoMa DINOv2 patch-divisibility assertion seen when
  generic `/32` preprocessing produced a 512-pixel dimension.
- Corrected LoMa match visualizations to plot normalized keypoints in the displayed processed
  image coordinates, including the Vismatch half-pixel convention; raw source dimensions are
  no longer used unless raw images are displayed.
- Aligned the Hydra default-composition regression test with the shipped `vismatch`
  public method and `rdd-lightglue` matcher defaults, while preserving `wildfusion`
  and LoMa as independently selectable methods.
- Fixed WildFusion probe finalization by importing the reporting `file_identity`
  helper; this prevents a completed matching run from failing during manifest assembly.
- Added a regression test that verifies the probe runner retains this reporting import.

### Known Issues

- Decided on 2026-08-17 to record the results of a second audit pass as known issues and
  defer the fixes. Full symptoms, reproductions, and measured impact are under
  "Known issues (open)" in AGENTS.md; the actionable items are also on the future-work
  checklist. Summary of what is open:
  - `scripts/summarize_runs.py --sort-by` raises `TypeError` on the shipped
    `reports/runs.csv`, because `sort_run_rows` mixes float and string sort keys and 11 of
    23 rows have an empty `top_1`. The command documented in README fails today.
  - The same sorter leaves rows unordered when a value is `NaN` and reports no error, which
    became reachable once `mAP` started reporting `nan` for shortlist-constrained methods.
  - Both probes allocate an unused `11924 x 27836` similarity matrix each epoch, and
    recompute full image-level retrieval metrics each epoch: about 1 minute and 2.7 GB of
    transient allocation per epoch, roughly 50 minutes at the shipped `epochs: 50`. This
    was previously unreachable because the probes crashed at epoch 1.
  - Probe per-epoch validation runs on the query/test split. Reported metrics are
    unaffected because they come from the post-loop final-epoch evaluation with no
    best-epoch selection, but per-epoch test curves invite selection on test.
  - Vismatch qualitative top-1 selection lacks an all-unscored guard;
    `_predict_class_probabilities` runs under grad during probe training; and
    `_to_hwc_uint8` would blacken float images if a transform ever preceded the dataset view.
- New visualization contact sheets require Pillow-readable image files and are omitted when no images are available.
- Historical `results/`, `benchmark_runs/`, and top-level visualization artifacts are intentionally not migrated.
- Probe and finetune no longer accept the legacy argparse flags or `--config`; existing scripts must use Hydra dotlist overrides.
- Hydra multirun sweeps are not supported by the project experiment contract.
- Full experiments require heavyweight dependencies, model downloads, and often
  CUDA-capable hardware.
- Custom LoMa checkpoints must match the selected LoMa architecture; the provided RDD-LightGlue checkpoint cannot be used for LoMa.
- Vismatch downloads matcher-specific weights on first use and can be expensive for
  large datasets; wrapped model licenses must be checked individually.
- Default dataset paths are environment-specific; the shared ex-reid environment must
  contain an importable pinned Vismatch installation.
- Masking and RDD improvements are dataset-dependent and must be validated per
  animal/dataset.
- Integration coverage for real backbones, CUDA, Vismatch weights, and private Lynx
  parity is intentionally outside the lightweight unit-test suite.
- The full RDD/LightGlue checkpoint and retrieval parity gate against the external
  Lynx fine-tuning implementation remains environment-gated and requires rerunning
  both default and custom checkpoints at `resize_max=512`.
- Finetuning still uses the current `test`/validation selection behavior; this
  implementation records the limitation rather than changing the protocol.
- Same-set WildFusion calibration excludes diagonal self-pairs by default and
  warns when no disjoint calibration split is configured; official all-pairs
  calibration remains an explicit compatibility mode.
- SHA-256 cache and split checks add I/O cost, and real Vismatch/WildFusion
  parity remains environment-gated.
- The 2026-08-14 code review found that configured WildFusion calibration is not yet
  truly disjoint, because one selected calibration dataset is passed to both sides.
- Mask metadata/content is not yet part of feature-cache fingerprints; changing a
  mask at the same image path can reuse stale features.
- Nested historical checkpoints without manifests are not discovered when automatic
  search starts at the repository-level `results/` root.
- Visualization and Vismatch qualitative rankings still have raw `argsort`/`argmax`
  paths that can disagree on ties with reported metrics.
- Concurrent jobs can race while updating `reports/runs.csv`, and manifests record
  checkpoint size/mtime but not SHA-256 or repository dirty-state identity.


### Added

- Added a pinned Vismatch matcher registry and feature-level backend for
  RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue.
- Added dependency-light tests for matcher validation, profile fingerprints, score
  normalization, empty matches, and Vismatch configuration defaults.
- Added canonical `FrameFeatures` shape/convention validation and an explicit
  migration error for the removed public `rdd` probe method.
- Kept the feature contract and profile registry dependency-light so base-environment
  tests do not import masking, CUDA, or downloaded matcher dependencies.
- Added Vismatch LoMa-B as the `loma` matcher profile, including its original
  14-pixel divisibility preprocessing, normalized-coordinate cache fields,
  mutual-match filtering, and profile-specific threshold default; the later
  Lynx-parity migration standardizes current Vismatch inputs on `/32` dimensions.

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
