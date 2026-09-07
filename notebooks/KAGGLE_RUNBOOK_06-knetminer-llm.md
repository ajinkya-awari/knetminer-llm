# Project 06 Kaggle runbook

[README](../README.md) | [Architecture](../docs/architecture.md) |
[Data and safety](../docs/data-and-safety.md) |
[Private Kaggle notebook](https://www.kaggle.com/code/ajinkya1225/06-knetminer-llm-validation)

Notebook: `kaggle_run_06-knetminer-llm.ipynb`

The checked-in notebook is an unexecuted, fail-closed template. Approval literals remain
`NOT_APPROVED`. For an authorized run, create a private staging copy, change only the required
approval literals there, and keep generated evidence outside the public source export.

## Validated Reference Run

Private notebook V8 completed on 2026-09-07 with Python 3.12.13, Torch 2.5.1+cu121, PyG 2.7.0,
CUDA 12.1, and a Tesla P100. It ran HGT seeds 42/43/44 and the 30-answerable/15-unanswerable model
gate with explicit CUDA placement. The frozen bundle passed its strict loader and artifact-manifest
checks. Strict Qwen acceptance remained 0/30, so this is execution and fallback-safety evidence, not
generated-answer-quality evidence. See the README for the complete V1-V8 history and measured
aggregate metrics.

## Required inputs

Attach one private Kaggle dataset containing the public source tree plus these approved private
inputs:

- `data/staging/opentargets_26.06_bounded_sample.json`
- `data/staging/real_structural_evaluation.json`

The notebook discovers the source root by checking `pyproject.toml`, `src/knetminer_llm`, and
`tests`. It rejects missing or ambiguous roots. Do not attach credentials, model weights, patient
data, raw provider responses, or unrelated project files.

The notebook installs `requirements-kaggle.txt`, which pins a matching PyTorch 2.5.1, TorchVision,
and TorchAudio CUDA 12.1 set for the Tesla P100 (`sm_60`) workers observed during validation. The
runtime also checks that the installed wheel contains a compatible architecture before starting HGT
work.

## Execution order

Run the 16 cells in order. Stop after a failed prerequisite.

1. Record safe configuration and blockers.
2. Discover exactly one attached Project 06 source tree.
3. Copy it to `/kaggle/working/06-knetminer-llm`, preserving an existing copy as a timestamped backup.
4. Validate the working tree, dependency manifest, staged snapshot, and 30/15 question artifact.
5. Install `requirements-kaggle.txt` only with `APPROVED_PROJECT_06_DEPENDENCIES`.
6. Restart the runtime once if the install changed imported packages.
7. Configure a provider-free synthetic test environment.
8. Run `python -m pytest -q` and store sanitized test evidence.
9. Display only the latest sanitized synthetic evidence.
10. Keep secret access disabled; the benchmark and local model need no token after internet download is approved.
11. Optionally load a named Kaggle Secret without printing its value.
12. Run the bounded Open Targets release-metadata probe only with `APPROVED_PROJECT_06_LIVE_PREFLIGHT`.
13. Run the synthetic HGT unit smoke only with `APPROVED_PROJECT_06_HGT_SMOKE`.
14. Run the exact seeds 42/43/44 CUDA HGT benchmark only with `APPROVED_PROJECT_06_HGT_BENCHMARK`.
15. Run the pinned Qwen 30-answerable/15-unanswerable evaluation and frozen bundle writer only with `APPROVED_PROJECT_06_MODEL_BUNDLE`.
16. Print the sanitized evidence inventory.

## Runtime commands

The HGT gate invokes:

```bash
python -m knetminer_llm.evaluation.cli hgt \
  --snapshot data/staging/opentargets_26.06_bounded_sample.json \
  --snapshot-sha 7a094e7ff9caa7cbd9b3c846ec75dffd985cefd18cca1afd242eafee9b6e7c51 \
  --minilm-dir /kaggle/working/models/all-MiniLM-L6-v2 \
  --output results/kaggle_evidence/hgt_benchmark.json \
  --device cuda --max-epochs 100 --patience 10 --allow-model-download
```

The model-backed gate invokes:

```bash
python -m knetminer_llm.evaluation.cli model-bundle \
  --snapshot data/staging/opentargets_26.06_bounded_sample.json \
  --snapshot-sha 7a094e7ff9caa7cbd9b3c846ec75dffd985cefd18cca1afd242eafee9b6e7c51 \
  --questions data/staging/real_structural_evaluation.json \
  --qwen-dir /kaggle/working/models/qwen2.5-1.5b-instruct \
  --output-dir results/kaggle_evidence/frozen_result_bundle \
  --device cuda \
  --allow-model-download
```

Model downloads are pinned to immutable revisions. HGT and the Kaggle model-bundle route must report
`requested_device=cuda` and `device_used=cuda`; CPU fallback is forbidden for either CUDA request.

## Evidence handling

Evidence is written below:

`/kaggle/working/06-knetminer-llm/results/kaggle_evidence/`

Download outputs to a private Project 06 directory. Verify every manifest hash before accepting a
result. Do not copy models, datasets, caches, raw generations, or private result bundles into the
public GitHub repository. A failed or partial output is not a release bundle.

The runtime may report measured HGT or model-backed results only after the corresponding output and
hashes are downloaded and validated. A release-metadata check or synthetic smoke is not a training
result.

## Acceptance Checklist

- Record Python, OS, Torch, PyG, CUDA, GPU name, requested device, and device used.
- Record seeds, sample counts, input identities, output identities, and every SHA-256.
- Require the exact 30-answerable/15-unanswerable artifact for the model gate.
- Validate citations, abstentions, bundle schema, trusted snapshot identity, and manifest hashes.
- Keep raw generations, source snapshots, models, and private bundles outside the public export.
- Preserve failed and cancelled versions as historical evidence; never relabel a partial directory
  as a release bundle.
