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
- [ ] Route visualization rankings and Vismatch qualitative top-1 selection through
  the shared stable ranking helper.
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
The `FrameFeatures` contract and matcher profiles remain dependency-light so unit tests can run without Vismatch, CUDA, downloaded weights, or masking packages. LoMa uses normalized[-1,1] keypoints internally, right/bottom padding to multiples of 14, and a default mutual-match threshold of 0.10; its feature cache records the original and padded image sizes.
Standard and Vismatch feature-cache fingerprints also include the resolved dataset root, metadata file, and `dataset.image_variant`; normal and pre-masked features must never share a cache identity. Mask payload/content is not yet hashed and remains future work.
Deep-feature cache keys must be constructed only after dataset and model-weight fingerprints are resolved; changing model contents must invalidate the key.
The 2026-08-12 full-split parity run found identical keypoint counts and descriptor
shapes but non-bit-identical feature tensors; mean absolute per-pair score difference
was 4.38e-05. The only ranking disagreement was a near-tie, so this result supports
behavioral equivalence but does not establish strict numerical identity.
The shipped probe YAML may intentionally select another Stage-A method (currently wildfusion); this does not disable the independently selectable `vismatch` method.
WildFusion uses `B` for candidate pairs per query, `local_batch_size` for pair-processing batches, and `local_top_k` for the ALIKED local keypoint budget. `local_top_k` defaults to 512 with `force_num_keypoints=True`; it is included in WildFusion cache/experiment identity so changing it does not reuse a different local-feature configuration.
Custom Vismatch checkpoints are selected with `benchmark.methods.vismatch.checkpoint_source`, `checkpoint_path`, and `checkpoint_components`. `default` preserves Vismatch-managed weights; `custom` accepts an exact model file or epoch directory. Component discovery uses tensor schemas and optional `checkpoint_manifest.json`, never filename ordering. RDD-LightGlue can load custom `rdd_extractor` and/or `lightglue` components, falling back to the default component in `auto` mode when one is absent. LoMa requires a validated LoMa-compatible checkpoint and explicit `loma_arch`; generic RDD/LightGlue files are rejected. Optimizer, scheduler, and random-state files are never loaded for probing. Component SHA-256 identities are part of Vismatch feature-cache keys and run manifests.
The production batching defaults are `batch_mode: batched`, `match_batch_size: 16`,
Matching displays a pair-counted tqdm progress bar with percentage, throughput, and ETA; progress advances only after successful batches, including after OOM retries.
and `extract_batch_size: 8`; `batch_mode: serial` remains the diagnostic/reference
workflow for parity checks. Extraction buckets images by matcher-native spatial shape.
Feature matching buckets candidate pairs across queries by exact left/right keypoint
cardinality and falls back to serial for empty, singleton, or otherwise incompatible
groups. LoMa is never
naively padded because padding would change assignment-softmax normalization. When
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
  and classifier probes; visualization ranking still requires migration to the helper.
- Primary `mAP` includes every query; a query with no relevant gallery identity contributes
  AP=0. `mAP_eligible` is the legacy eligible-query-only diagnostic, and coverage fields
  report how many queries had a gallery match.
- Linear and efficient probes report identity-level metrics as primary. Their existing
  image-level matrix and metrics remain under `image_*` diagnostic fields.
- Vismatch and WildFusion use shortlist-constrained ranking. Vismatch overwrites only
  shortlisted candidates in a matrix initialized to `-inf`; invalid candidate scores
  also become `-inf`. It reports candidate hit/recall and scored/unscored pair counts.
  The previous finite Stage-A fallback mixed incompatible cosine and matcher score
  scales and is not a supported production policy.
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
