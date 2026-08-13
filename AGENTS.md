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
- config/: YAML experiment configuration.
- train/ and scripts/: command-line entrypoints.
- tests/: dependency-light regression tests.
- results/, benchmark_runs/, kaggle_runs/, cache/, and visualizations/:
  generated artifacts; do not edit them manually.

## Standard commands

Run from the repository root:

~~~bash
python train/finetune.py --config config/finetune_config.yaml
python train/probe.py --config config/probe_config.yaml
python scripts/kaggle_jaguar_submit.py --config config/kaggle_jaguar.yaml --data-dir /path/to/jaguar-re-id
python -m unittest discover -s tests -p 'test_*.py'
python -m py_compile models/*.py reid/**/*.py train/*.py scripts/*.py
~~~

Use --dry-run, --pair-limit, or --fast for Jaguar development runs.
Do not run full GPU training or Vismatch benchmarks as a default validation step.

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
- Record assumptions, compatibility decisions, and unresolved issues in the handoff.

## External environment

The code depends on PyTorch/torchvision, timm, Hugging Face Transformers,
wildlife-datasets, wildlife-tools, pycocotools, OpenCV, and other packages listed
in requirements.txt. Backbone weights may require network access on first use.
Vismatch is pinned to commit 4a743b75749a3770af59d275483ed341dea51ff0 and downloads matcher weights on first use. The shared ex-reid environment must have an importable, non-broken Vismatch installation; it must not depend on a missing editable checkout.
Default paths are specific to the original shared compute environment.

## Future-work checklist

- [ ] Add optional integration tests with a fake/local backbone and synthetic images.
- [ ] Add CI for unit tests, syntax checks, and YAML/config validation.
- [ ] Replace environment-specific absolute paths with machine-local overrides.
- [ ] Pin external Git dependencies to reproducible commits.
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
- [ ] Reconcile historical experiment metadata and stale generated CSV schemas.
- [ ] Expand experiment reports with identity counts and protocol summaries.


## Vismatch matcher policy

The public local-matcher method is `vismatch`; the selected matcher is configured
under `benchmark.methods.vismatch.matcher`. Supported initial profiles are
`rdd-lightglue`, `aliked-lightglue`, `superpoint-lightglue`, and `loma` (Vismatch's
LoMa-B wrapper). The production path extracts features once and matches cached features. `feature_matching_mode: feature_level`
is required for production; pairwise Vismatch calls are reserved for explicit diagnostics.
Old `rdd` method names and direct RDD repository paths are unsupported and receive a migration-specific error.
The `FrameFeatures` contract and matcher profiles remain dependency-light so unit tests can run without Vismatch, CUDA, downloaded weights, or masking packages. LoMa uses normalized[-1,1] keypoints internally, right/bottom padding to multiples of 14, and a default mutual-match threshold of 0.10; its feature cache records the original and padded image sizes.
The 2026-08-12 full-split parity run found identical keypoint counts and descriptor
shapes but non-bit-identical feature tensors; mean absolute per-pair score difference
was 4.38e-05. The only ranking disagreement was a near-tie, so this result supports
behavioral equivalence but does not establish strict numerical identity.
The shipped probe YAML may intentionally select another Stage-A method (currently wildfusion); this does not disable the independently selectable `vismatch` method.
The production batching defaults are `batch_mode: batched`, `match_batch_size: 16`,
and `extract_batch_size: 8`; `batch_mode: serial` remains the diagnostic/reference
workflow for parity checks. Extraction buckets images by matcher-native spatial shape.
Feature matching buckets candidate pairs across queries by exact left/right keypoint
cardinality and falls back to serial for empty, singleton, or otherwise incompatible
groups. LoMa is never
naively padded because padding would change assignment-softmax normalization. When
CUDA runs out of memory and `oom_backoff: true`, the current batch is retried at half
size, temporary CUDA memory is cleared, and effective batch sizes are recorded in
Vismatch timing metadata. Stage-A candidates, `candidate_k`, matrix placement, and
score normalization are unchanged; the 1e-4 score/top-1 parity gate remains required
before interpreting performance results.
