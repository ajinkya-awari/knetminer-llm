# Data and Safety

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
