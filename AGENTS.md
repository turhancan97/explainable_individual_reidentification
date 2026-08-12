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
  efficient probe, and RDD benchmark dispatch.
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
Do not run full GPU training or RDD benchmarks as a default validation step.

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
  external model/RDD behavior while fixing infrastructure issues.
- Record assumptions, compatibility decisions, and unresolved issues in the handoff.

## External environment

The code depends on PyTorch/torchvision, timm, Hugging Face Transformers,
wildlife-datasets, wildlife-tools, pycocotools, OpenCV, and other packages listed
in requirements.txt. Backbone weights may require network access on first use.
RDD additionally requires a separate local RDD checkout and RDD/LightGlue weights.
Default paths are specific to the original shared compute environment.

## Future-work checklist

- [ ] Add optional integration tests with a fake/local backbone and synthetic images.
- [ ] Add CI for unit tests, syntax checks, and YAML/config validation.
- [ ] Replace environment-specific absolute paths with machine-local overrides.
- [ ] Pin external Git dependencies to reproducible commits.
- [ ] Evaluate masking and RDD settings separately for each animal dataset.
- [ ] Profile and optimize RDD feature extraction/reranking costs.
- [ ] Consider atomic checkpoint writes and explicit checkpoint retention.
- [ ] Reconcile historical experiment metadata and stale generated CSV schemas.
- [ ] Expand experiment reports with identity counts and protocol summaries.
