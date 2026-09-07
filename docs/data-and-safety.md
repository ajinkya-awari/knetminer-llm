# Data and Safety

[README](../README.md) | [Architecture](architecture.md) |
[Evidence workbench](demo/) | [Third-party notices](../THIRD_PARTY_NOTICES.md)

The intended source is Open Targets Platform release 26.06 under CC0. A runtime snapshot must preserve
the release, retrieval time, source URL, request hash, response hash, stable IDs, normalized relation,
score when supplied, and provenance identifiers. This source repository intentionally includes no
snapshot or generated evaluation result.

The package accepts exactly three node types and three citable forward relations. Labels are display
attributes and never replace stable identifiers. Reverse graph edges are non-citable transport records.
A failed release check, malformed GraphQL response, unknown endpoint, duplicate identity, provenance
conflict, graph mismatch, split leakage, or bundle mismatch is a hard failure.

The project is for evidence-retrieval research and is not for clinical use. It does not diagnose,
recommend treatment, estimate patient risk, or provide medical advice. Do not process patient data,
credentials, restricted sources, or account-gated content. Model-generated text is untrusted until it
passes the exact output, claim-text, evidence-path, and citation checks.

## Public Release Boundary

| Included | Excluded |
| --- | --- |
| Source code, tests, schemas, and public documentation | Open Targets source snapshots |
| Unexecuted fail-closed Kaggle notebook and runbook | Credentials, tokens, and private Kaggle metadata |
| Sanitized CC0-derived stable IDs and forward edges used by the static demo | Model weights, checkpoints, embeddings, and caches |
| Measured aggregate metrics with provenance and hashes | Raw prompts, generations, and private result bundles |

The static bundle is data, not executable authority. Its cited path and edge IDs must exist in the
bundle inventory, and the loader must match the caller-supplied trusted snapshot hash before any
record is rendered.

## Claims Boundary

The validated V8 run supports claims about execution, artifact integrity, citation validity, and
abstention behavior on the bounded evaluation. It does not support clinical use, biological
discovery, broad benchmark generality, HGT superiority, or generated-answer quality. In particular,
0/30 Qwen outputs passed the exact validator; the published answerable rows use deterministic,
evidence-backed fallbacks.
