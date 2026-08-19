# AGENTS.md

## Project purpose

This repository trains and evaluates wildlife individual re-identification systems.
Given query and database images, it produces identity-retrieval similarity scores.
The primary datasets are wildlife datasets such as CzechLynx, LeopardID2022,
ReunionTurtles, and Kaggle Jaguar data.

The project is retrieval-oriented. Do not describe or evaluate it as a conventional
single-label animal-species classifier: identity labels represent individual animals.

## Repository map

- models/model.py: pretrained backbone factory and ViT CLS adapter.
- models/objective.py: ArcFace, triplet, softmax, and efficient-probe objectives.
- reid/engine/finetune_runner.py: ArcFace finetuning and validation retrieval.
- reid/engine/probe_runner.py: cosine, WildFusion, local LightGlue, linear probe,
  efficient probe, and Vismatch matcher benchmark dispatch.
- reid/engine/kaggle_jaguar_runner.py: standalone Jaguar workflow.
- reid/data/: dataset views, COCO-RLE masking, and split safety checks.
- reid/evaluation/metrics.py: top-k, balanced top-1, and mAP calculations.
- reid/training/: checkpoint serialization and accumulation helpers.
- reid/reporting/: run identities, manifests, metrics, visualization indexes, and summaries.
- conf/: Hydra configuration for probe and finetuning.
- config/: standalone Jaguar YAML configuration and other non-Hydra configs.
- train/ and scripts/: command-line entrypoints.
- tests/: dependency-light regression tests.
- experiments/, reports/, results/, benchmark_runs/, kaggle_runs/, cache/, and visualizations/:
  generated artifacts; do not edit them manually.

## Standard commands

Run from the repository root:

~~~bash
python train/finetune.py
python train/finetune.py train.epochs=10
python train/probe.py
python train/probe.py benchmark.method=vismatch benchmark.methods.vismatch.matcher=loma
bash probe-parallel.sh --list-tasks
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id
python scripts/summarize_runs.py --format markdown
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile models/*.py reid/**/*.py train/*.py scripts/*.py
~~~

Use --dry-run, --pair-limit, or --fast for Jaguar development runs.
Do not run full GPU training or Vismatch benchmarks as a default validation step.

## Hydra configuration

`train/probe.py` and `train/finetune.py` use Hydra 1.3 as their primary single-run
configuration interface. Defaults live in `conf/probe.yaml` and `conf/finetune.yaml`;
use nested dotlist overrides such as `benchmark.method=vismatch` or
`train.epochs=10`. Hydra/OmegaConf performs type conversion and rejects unknown or
misspelled configuration paths. The legacy argparse flags and `--config` option are
not supported by these entrypoints.

Hydra does not change the working directory. New probe and finetune runs use the
reporting-managed `experiments/` layout while legacy aggregate CSVs remain under
`benchmark_runs/` and `results/`. Hydra multirun sweeps are intentionally outside
the current experiment contract. Jaguar remains on its
existing argparse and `config/kaggle_jaguar.yaml` workflow; its internal finetune
template points to `conf/finetune.yaml`.

`probe-parallel.sh` is a separate self-submitting Slurm launcher and must not
modify or replace `probe.sh`. It builds an explicit 54-task grid: nine method or
checkpoint variants (cosine, WildFusion, local LightGlue, linear probe, efficient
probe, default/custom LoMa, and default/custom RDD-LightGlue) crossed with
`candidate_k` values `10, 50, 100, 250, 500, 1000`. Run it with
`bash probe-parallel.sh`; `MAX_CONCURRENT_JOBS` defaults to `4` and becomes the
Slurm array `%` throttle. `--list-tasks` and `PROBE_PARALLEL_DRY_RUN=1` are safe
non-executing inspection modes. The custom checkpoint variables are defined near
the top of the launcher, and custom Vismatch tasks explicitly use
`checkpoint_components=matcher_only`. The launcher validates both custom paths
before submission and prints the complete Hydra command in each task log.

## Experiment artifacts

Modern probe and finetune runs use `experiments/` and are self-contained. Paths are
organized as dataset/animal/split/model/method/variant/run-id. Each run must retain
`config.snapshot.yaml`, `run_manifest.json`, `metrics.json`, and `timings.json`;
finetune runs also retain `training_metrics.csv` and canonical checkpoints.

`reports/runs.csv` is the central one-row-per-run index. Legacy
`benchmark_runs/benchmark_results.csv` and `results/.../train_metrics.csv` remain
populated for compatibility. Historical generated artifacts are never migrated or
rewritten automatically.

New visualizations belong inside the run’s `visualizations/` directory. Their
`index.csv` must map query/database identities, ranks, scores, correctness, and
artifact paths. Top-1 and failure contact sheets are optional when no images or
labels are available. Use `scripts/summarize_runs.py` for filtered Markdown/CSV
comparisons.

Run directories use UTC timestamps plus a short resolved-configuration hash. Do not
reuse a run directory or manually edit manifests, indexes, checkpoints, or generated
images. Explicit checkpoint paths take precedence; automatic discovery searches
`experiments/finetune/` before legacy `results/` and never selects full checkpoints
for inference.

## Development Environment

Use the shared conda environment for repository work:

    /shared/results/common/kargin/tck_miniconda3/envs/ex-reid

Typical activation:

    source /shared/results/common/kargin/tck_miniconda3/etc/profile.d/conda.sh
    conda activate ex-reid

Do not create or switch to another environment unless the user explicitly requests
it or the shared environment is unavailable.

## Configuration and data contracts

Metadata CSVs must provide the configured identity and split columns. Image paths
must resolve relative to the configured dataset root unless absolute. When
no_background is true, the configured mask column must contain valid JSON COCO-RLE
data with dimensions matching the source image.

The shipped default model is megadescriptor-l. Supported identifiers are
megadescriptor-t, megadescriptor-l, lynx_megadescriptorV3,
lynx_megadescriptorV4, miewid, dinov2, and dinov3. The legacy identifier
megadescriptor is intentionally unsupported.

Classifier probes are closed-set methods: query identities must occur in the
database identities. Retrieval methods may evaluate unseen identities, but the
safety-check output must be considered before interpreting metrics.

## Checkpoints

Canonical finetune outputs are:

- checkpoint-final.pth
- checkpoint-final-full.pth
- checkpoint-latest-full.pth
- checkpoint-best.pth
- checkpoint-best-full.pth
- checkpoint-epoch-<n>.pth

Tagged model-only files such as checkpoint-final_<dataset_tag>.pth remain readable
for compatibility with historical runs. Explicit checkpoint paths take precedence;
automatic probe/Jaguar discovery searches the newest run for canonical model-only
files and then compatible tagged model-only final files. Never pass a *-full.pth
file to inference code expecting a model-only state dict.

## Implementation rules for AI sessions

- Inspect the working tree before editing and preserve unrelated user changes.
- Keep changes focused on the requested behavior; do not rewrite generated artifacts.
- Avoid destructive commands such as resets, broad recursive deletion, or overwriting
  unrelated results.
- Use patch-based edits when possible and fail closed when expected source text differs.
- Prefer lightweight, deterministic tests that do not download models or require CUDA.
- Run relevant unit tests and syntax checks before handoff.
- After every code, configuration, documentation, or workflow change, update both
  AGENTS.md and CHANGELOG.MD before handoff.
- After every important architectural, compatibility, experiment, or workflow
  decision, record the decision and its rationale in AGENTS.md and CHANGELOG.MD.
- Keep AGENTS.md current as the operating guide and future-work source of truth;
  keep CHANGELOG.MD as the chronological record of changes and decisions.
- Do not silently change benchmark protocols, score ranges, split semantics, or
  external matcher behavior while fixing infrastructure issues.
- Probe calibration must use the dataset returned by `load_dataset_splits`; failed-run
  reporting must preserve the original exception and create missing report parents.
- Probe finalization must import and use `file_identity` from `reid.reporting.artifacts`;
  final WildFusion matching must not be considered successful until manifest/report
  assembly completes.
- Console metric reporting must handle both numeric metrics and string diagnostic
  fields such as Vismatch cache fingerprints without changing persisted metric values.
- Record assumptions, compatibility decisions, and unresolved issues in the handoff.

## External environment

The code depends on PyTorch/torchvision, timm, Hugging Face Transformers,
wildlife-datasets, wildlife-tools, pycocotools, OpenCV, and other packages listed
in requirements.txt. Backbone weights may require network access on first use.
Hydra is pinned to `hydra-core==1.3.2`; Vismatch is pinned to commit 4a743b75749a3770af59d275483ed341dea51ff0 and downloads matcher weights on first use. The shared ex-reid environment must have an importable, non-broken Vismatch installation; it must not depend on a missing editable checkout.
Default paths are specific to the original shared compute environment.

## Known issues (open)

Audited on 2026-08-17 and deliberately deferred. Each entry records the symptom, a
reproduction, and the measured impact so it can be picked up without re-investigation.

- **`sort_run_rows` raises `TypeError` on the shipped run index.**
  `reid/reporting/summary.py:86` returns a float when a cell parses and a string when it
  does not, so a column that is numeric in some rows and empty in others cannot be sorted.
  `reports/runs.csv` currently has 23 rows, 11 with an empty `top_1` from failed runs.
  Reproduce with the documented command
  `python scripts/summarize_runs.py --sort-by top_1 --format markdown`, which fails today.
  Fix by coercing unparseable cells to a sentinel that orders consistently.

- **`sort_run_rows` silently mis-sorts when a value is `NaN`.**
  Same function. NaN comparisons are all false, so the sort leaves rows in place and
  returns output that looks sorted but is not: `[0.5, nan, 0.9, 0.1]` comes back
  unchanged. This became reachable when `mAP` started reporting `nan` for
  shortlist-constrained methods, so sorting runs by `mAP` now yields a meaningless order
  with no error. Sort NaN to the end explicitly.

- **Dead 1.33 GB allocation per probe epoch.**
  `reid/engine/probe_runner.py:913` and `:1129` assign `similarity_epoch` and never read
  it. At CzechLynx scale that is an `11924 x 27836` float32 matrix built and discarded
  every epoch in both `linear_probe` and `efficient_probe`. Delete the statement.

- **Per-epoch image-level metrics dominate probe runtime.**
  `_probe_retrieval_metrics` rebuilds the same `11924 x 27836` matrix and ranks it in full
  every epoch: measured about 1 minute and 2.7 GB of transient allocation per epoch, so
  roughly 50 minutes at the shipped `epochs: 50`. Previously invisible because both probes
  crashed at epoch 1. Restrict the per-epoch call to identity-level metrics and compute the
  `image_*` diagnostics once after training.

- **Probe per-epoch validation uses the query/test split.**
  `run_linear_probe` and `run_efficient_probe` build their `[*][val]` loader from
  `dataset_query`, logging test loss and test metrics every epoch. Reported metrics are not
  affected: they come from the post-loop evaluation of the final-epoch model, and no
  best-epoch selection occurs. The hazard is downstream, since per-epoch test curves in
  W&B invite epoch or hyperparameter selection on the test split. Same class of limitation
  as the finetune selection split.

- **Vismatch qualitative top-1 can point at an unscored pair.**
  When a query row is entirely `-inf`, `stable_rank_1d(...)[0]` returns database index 0
  and a meaningless match image is drawn instead of the query being skipped.

- **`_predict_class_probabilities` runs under grad during probe training.**
  `reid/engine/probe_runner.py:875` builds a graph for the softmax and then detaches it.
  Wasteful, not incorrect.

- **`_to_hwc_uint8` would destroy float images.**
  `reid/data/dataset_view.py:86` clips non-uint8 input to `{0, 1}` before masking. Not
  triggered today because the base dataset yields PIL images, but it would silently blacken
  inputs if a transform were ever applied before the view.

## Future-work checklist

- [ ] Add optional integration tests with a fake/local backbone and synthetic images.
- [ ] Add CI for unit tests, syntax checks, and YAML/config validation.
- [x] Migrate probe and finetuning configuration to Hydra with strict dotlist overrides and resolved snapshots.
- [ ] Replace environment-specific absolute paths with machine-local overrides.
- [ ] Pin external Git dependencies to reproducible commits.
- [ ] Implement truly disjoint calibration inputs for WildFusion and local matcher
  calibration; the current split setting selects one dataset and passes it to both
  sides of calibration.
- [ ] Include mask metadata/content fingerprints in standard and Vismatch feature
  caches so mask edits invalidate features, not only image-file edits.
- [x] Route visualization index rankings through the shared stable ranking helper;
  `visualizations/index.csv` now uses `stable_rank_1d`. Vismatch qualitative top-1
  selection still needs an all-unscored guard, tracked under Known issues.
- [ ] Make `sort_run_rows` total: coerce unparseable cells and order `NaN` last, so
  `scripts/summarize_runs.py --sort-by` works on mixed and gated metric columns.
- [ ] Reduce probe per-epoch metric cost: drop the unused `similarity_epoch` allocation
  and compute `image_*` diagnostics once after training rather than every epoch.
- [ ] Give the probes a validation split distinct from the query/test split, or stop
  logging per-epoch test metrics, so epoch and hyperparameter choices cannot use it.
- [ ] Make legacy checkpoint discovery recursive for the existing nested no-manifest
  `results/<dataset>/<animal>/mask_<...>/run_<...>` layout.
- [ ] Evaluate masking and Vismatch matcher settings separately for each animal dataset.
- [x] Fix runtime annotation import validation for the batched Vismatch path.
- [x] Add an ex-reid-gated runtime smoke test that invokes batched extraction.
- [x] Keep explicit batching defaults in the user-preserved probe YAML.
- [x] Add production-batched Vismatch feature extraction and feature-level reranking
  with a permanent serial parity/reference mode.
- [ ] Profile and optimize cached Vismatch feature extraction/reranking costs.
- [ ] Run the private Lynx golden-subset parity comparison for RDD-LightGlue before changing matcher defaults.
- [x] Run the available full-split Lynx parity comparison on 2026-08-12: 66 queries,
  217 gallery sequences, one frame per sequence; top-1 agreement was 65/66
  (98.48%), top-5 agreement was 100%, and mAP differed by -0.0018. The fixed
  golden-subset gate remains open because the comparison was not 100% top-1.
- [ ] Complete matcher ablations for RDD-LightGlue, ALIKED-LightGlue, SuperPoint-LightGlue, and LoMa-B.
- [ ] Track wrapped-model licenses and downloaded-weight provenance for paper release.
- [ ] Consider atomic checkpoint writes and explicit checkpoint retention.
- [x] Record completed finetuning total runtime in train_metrics.csv.
- [ ] Reconcile historical experiment metadata and stale generated CSV schemas.
- [x] Add readable experiment manifests, run indexing, visualization indexes, and summary reports.
- [ ] Make central run-index updates safe for concurrent jobs and use unique temporary
  files or locking instead of a shared `reports/runs.csv.tmp` path.
- [ ] Record SHA-256 checkpoint identities and repository dirty-state/diff identity in
  manifests so uncommitted experiment code remains reproducible.
- [x] Repair stale configuration expectations and synthetic-image fixtures so the full
  ex-reid regression suite is green before relying on it as a release gate.


## Vismatch matcher policy

The public local-matcher method is `vismatch`; the selected matcher is configured
under `benchmark.methods.vismatch.matcher`. Supported initial profiles are
`rdd-lightglue`, `aliked-lightglue`, `superpoint-lightglue`, and `loma` (Vismatch's
LoMa-B wrapper). The production path extracts features once and matches cached features. `feature_matching_mode: feature_level`
is required for production; pairwise Vismatch calls are reserved for explicit diagnostics.
Old `rdd` method names and direct RDD repository paths are unsupported and receive a migration-specific error.
The `FrameFeatures` contract and matcher profiles remain dependency-light so unit tests can run without Vismatch, CUDA, downloaded weights, or masking packages. RDD-LightGlue, ALIKED-LightGlue, and SuperPoint-LightGlue use the pinned Lynx-compatible preprocessing: RGB float32 tensors in `[0,1]`, direct bilinear tensor resize to the configured target long side, and floor of each dimension to a multiple of 32. LoMa uses the LoMa fine-tuning protocol with the same tensor interpolation but floors dimensions to a multiple of 14 for its DINOv2-L/14 descriptor. LoMa uses normalized[-1,1] keypoints and a default mutual-match threshold of 0.10; its feature cache records processed and original image sizes.
Standard and Vismatch feature-cache fingerprints also include the resolved dataset root, metadata file, and `dataset.image_variant`; normal and pre-masked features must never share a cache identity. Mask payload/content is not yet hashed and remains future work.
Deep-feature cache keys must be constructed only after dataset and model-weight fingerprints are resolved; changing model contents must invalidate the key.
The 2026-08-12 full-split parity run found identical keypoint counts and descriptor
shapes but non-bit-identical feature tensors; mean absolute per-pair score difference
was 4.38e-05. The only ranking disagreement was a near-tie, so this result supports
behavioral equivalence but does not establish strict numerical identity.
The shipped probe YAML may intentionally select another Stage-A method (currently wildfusion); this does not disable the independently selectable `vismatch` method.
WildFusion and Local LightGlue derive their refinement `B` from `benchmark.candidate_k`; `local_batch_size` controls pair-processing batches, and `local_top_k` controls the ALIKED local keypoint budget. `local_top_k` defaults to 512 with `force_num_keypoints=True`; it is included in WildFusion cache/experiment identity so changing it does not reuse a different local-feature configuration.
Custom Vismatch checkpoints are selected with `benchmark.methods.vismatch.checkpoint_source`, `checkpoint_path`, and `checkpoint_components`. `default` preserves Vismatch-managed weights; `custom` accepts an exact model file or epoch directory. Component discovery uses tensor schemas and optional `checkpoint_manifest.json`, never filename ordering. RDD-LightGlue can load custom `rdd_extractor` and/or `lightglue` components, falling back to the default component in `auto` mode when one is absent. LoMa requires a validated LoMa-compatible checkpoint and explicit `loma_arch`; generic RDD/LightGlue files are rejected. Optimizer, scheduler, and random-state files are never loaded for probing. Component SHA-256 identities are part of Vismatch feature-cache keys and run manifests.
The Vismatch `resize_max` field is the target long-side resolution, not a downscaling-only cap; the shipped default is 512. RDD-family Vismatch profiles use preprocessing identity `lynx_finetuning_v1` and `/32` dimensions. LoMa uses `lynx_loma_finetuning_v1` and `/14` dimensions. Changing the preprocessing identity or target resolution invalidates Vismatch feature caches. Cosine, WildFusion, local LightGlue, linear probe, and efficient probe retain their existing square-resize protocols.
LoMa match visualizations must use the processed-image coordinate space shown on the canvas: convert normalized keypoints to `FrameFeatures.image_size` coordinates and apply the Vismatch/LoMa half-pixel convention, without scaling points back to `original_image_size` unless the visualization also displays raw images.
The production batching defaults are `batch_mode: batched`, `match_batch_size: 16`,
Matching displays a pair-counted tqdm progress bar with percentage, throughput, and ETA; progress advances only after successful batches, including after OOM retries.
and `extract_batch_size: 8`; `batch_mode: serial` remains the diagnostic/reference
workflow for parity checks. Extraction buckets images by matcher-native spatial shape.
Feature matching buckets candidate pairs across queries by exact left/right keypoint
cardinality and falls back to serial for empty, singleton, or otherwise incompatible
groups. No matcher input is padded to a common keypoint count; LoMa is therefore not
subjected to padding that would change assignment-softmax normalization. When
CUDA runs out of memory and `oom_backoff: true`, the current batch is retried at half
size, temporary CUDA memory is cleared, and effective batch sizes are recorded in
Vismatch timing metadata. Stage-A candidates and `candidate_k` remain unchanged. Vismatch
and WildFusion are shortlist-constrained: Vismatch initializes unscored positions to
`-inf`, matching WildFusion's sparse matrix behavior. The policy is recorded as
`score_matrix_policy=shortlist_only_neg_inf` with candidate/unscored pair counts and
candidate fraction. The 1e-4 score/top-1 parity gate remains required before interpreting
performance results.
The explicit `dataset.image_variant` field must be `background` or `no_background`.
Use `background` for normal `images/` inputs and `no_background` for pre-masked
`masked_images/` inputs or dynamically masked images. This field is provenance,
separate from `dataset.no_background`, which controls whether an RLE mask is
applied at load time.

## Research-validity reporting policy

- Primary retrieval metrics use deterministic descending scores with original database
  index as the tie-breaker. This rule is shared by evaluation, shortlisting, Jaguar,
  and classifier probes. Visualization ranking is now migrated: the run-local
  `visualizations/index.csv` uses `stable_rank_1d`, so it resolves ties identically to the
  prediction grid it annotates and to the metrics. No ranking path may use
  `argsort()[::-1]`, which reverses a stable ascending sort and orders ties backwards.
- Primary `mAP` includes every query; a query with no relevant gallery identity contributes
  AP=0. `mAP_eligible` is the legacy eligible-query-only diagnostic, and coverage fields
  report how many queries had a gallery match.
- `mAP` and `mAP_eligible` are emitted only when `score_coverage == 1.0`, that is when every
  matrix position carries a real score. A shortlist matrix leaves ~99.6% of each row at
  `-inf` ordered by original database index, so a full-matrix mAP there measures metadata
  row adjacency rather than the method: Vismatch runs spanning `top_1` 0.360-0.412 all
  produced `mAP` in 0.0463-0.0469, against 0.0168 for a random ranking. Both fields become
  `nan` instead, and no un-gated variant is persisted, so the number cannot re-enter a
  comparison by accident.
- `mAP_at_k` is the primary metric for shortlist methods and is computed identically for
  full-matrix methods, so cosine, WildFusion, and Vismatch stay comparable. It truncates at
  `benchmark.candidate_k`, grants no credit to positions the method never scored, and divides
  by `min(relevant, k)` so a shortlist miss scores 0. `rerank_mAP_at_k` divides instead by
  the hits present in the scored top-k and isolates Stage-B ordering from Stage-A reach;
  read it together with `recall_at_k` and `candidate_recall_at_k`.
- Evaluation cutoffs are validated before model loading: every `benchmark.top_k` entry and
  `benchmark.candidate_k` must fit inside the Vismatch shortlist. Vismatch candidate
  selection, WildFusion refinement (`B`), and the mAP@k cutoff all derive from this one
  setting. The old independent budget overrides, including Local LightGlue `B`, are rejected; use `benchmark.candidate_k`.
- Probe runs persist finite score-matrix entries to run-local `scores.npz` in sparse COO
  form. Metric definitions can then be revised without repeating a matcher run. Dense
  matrices above the entry budget are skipped rather than written.
- Linear and efficient probes report identity-level metrics as primary. Their existing
  image-level matrix and metrics remain under `image_*` diagnostic fields.
- Identity-level probe scores come from the classifier's per-identity output columns, not
  from a per-database-image maximum. The head emits one probability per identity, so all
  images of an identity share it and any maximum over them is a no-op; masking the class
  axis with an image-length mask raised `IndexError` and blocked both probes entirely.
  Database label indices outside the classifier head are rejected rather than silently
  reindexed. Probe classification results remain closed-set and are not directly
  comparable to the retrieval methods.
- Vismatch and WildFusion use shortlist-constrained ranking. Vismatch overwrites only
  shortlisted candidates in a matrix initialized to `-inf`; invalid candidate scores
  also become `-inf`. It reports candidate hit/recall and scored/unscored pair counts.
  The previous finite Stage-A fallback mixed incompatible cosine and matcher score
  scales and is not a supported production policy.
- File digests are memoized only inside an explicit `file_digest_cache()` block, which
  `run_probe` and `run_finetune` wrap around a whole run. The block asserts that the files
  being hashed are stable for its duration. Never make this memoization process-wide: tmpfs
  reuses one `st_mtime_ns` for rapid same-size rewrites, so a global cache could serve a
  stale digest and silently break content-addressed cache identities. Entries are keyed on
  device/inode/size/mtime so the safety-check, dataset-digest, and Vismatch cache-key paths
  share them despite constructing paths differently.
- Unseen query identities are always reported by split safety checks. `require_b_labels_in_a`
  selects the response: closed-set classifier probes fail, open-set retrieval warns and
  continues. There is no warn-while-required mode; the old `warn_only_unseen` flag was
  unreachable and has been removed.
- Vismatch preprocessing runs exactly once per image. `prepare_image()` returns a
  `PreparedImage` (tensor plus source and processed sizes) that supplies the batch-bucketing
  shape and is consumed directly by `extract_prepared`/`extract_prepared_batch`. Do not
  reintroduce a shape probe that re-runs the resize, and keep buckets holding prepared
  tensors rather than full-resolution sources. Any change here must keep extracted features
  bit-identical, since feature-cache identities do not cover this code path.
- Vismatch merges its Stage-A metrics under a `stage_a_<method>_` prefix after
  `run_vismatch_benchmark` returns, since that call replaces the metrics dict. Do not coerce
  merged metric values to float: some are string diagnostics.
- Split safety preserves path-overlap checks and additionally hashes resolved files with
  SHA-256. Duplicate content across protected splits fails closed and is summarized with
  sample paths and unreadable-file counts.
- Standard and Vismatch feature caches include image-content SHA-256, preprocessing,
  metadata, image variant, model/checkpoint identity, and matcher profile/weight identity,
  but do not yet include mask metadata/content hashes. Image contents at a fixed path
  invalidate caches; mask edits require the future cache-fingerprint fix.
- Automatic inference checkpoint discovery searches recursively under modern finetune
  experiments, ignores failed/incomplete manifests and `*-full.pth`, prefers completed
  canonical model-only files, then tagged legacy files, and preserves explicit-path priority.
  Existing nested legacy runs without manifests are not yet discovered when searching
  from the repository-level `results/` root.
- Finetune resume fails closed when the checkpoint leaves no epochs to run
  (`start_epoch >= train.epochs`). A zero-epoch run would still write final checkpoints and
  a completed manifest, hiding an unraised `train.epochs`; the guard runs before training
  setup so nothing is written.
- Finetune reports select and reload the best model-only checkpoint for primary metrics;
  final-epoch metrics remain nested as `final_epoch_metrics`. The current test/validation
  split remains the selection split and is a documented limitation.
- Accumulation divides each raw loss by the actual microbatch count in its group, including
  a partial final group; optimizer-step boundaries and scheduler behavior are unchanged.
- WildFusion calibration excludes same-image diagonal pairs by default and warns on the
  database-derived fallback. The configured split currently produces one calibration
  dataset used on both sides, so a truly disjoint calibration protocol remains future work.
  `official_same_set: true` enables exact all-pairs compatibility calibration for parity.

These validity changes are forward-only. Historical generated artifacts, aggregate CSVs,
and old caches are not rewritten automatically; rerun affected experiments before using
