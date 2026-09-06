from __future__ import annotations

import hashlib
import json
from pathlib import Path

from knetminer_llm.app import FrozenDemoBundle
from knetminer_llm.data.normalize import NormalizedSnapshot
from knetminer_llm.evaluation.runtime import canonical_sha256


def write_frozen_result_bundle(
    snapshot: NormalizedSnapshot,
    evaluation: dict[str, object],
    *,
    snapshot_hash: str,
    output_dir: Path,
    model_revision: str,
) -> dict[str, object]:
    """Write and validate canonical public-result artifacts and their hashes."""

    if len(snapshot_hash) != 64:
        raise ValueError("snapshot hash must be SHA-256 hex")
    summary = evaluation["summary"]
    rows = evaluation["rows"]
    evidence_paths = evaluation["evidence_paths"]
    if not isinstance(summary, dict) or not isinstance(rows, list) or not isinstance(evidence_paths, dict):
        raise ValueError("evaluation result shape is invalid")
    if summary.get("citation_validity") != 1.0 or float(summary.get("abstention_f1", 0.0)) < 0.90:
        raise ValueError("evaluation does not satisfy citation and abstention release gates")
    evaluation_hash = canonical_sha256({"summary": summary, "rows": rows})
    bundle_payload = {
        "schema_version": "1",
        "provenance": {
            "source_release": snapshot.manifest.source_release,
            "licence": snapshot.manifest.licence,
            "source_url": str(snapshot.manifest.source_url),
            "request_hash": snapshot.manifest.request_hash,
            "response_hash": snapshot.manifest.response_hash,
            "snapshot_hash": snapshot_hash.lower(),
            "artifact_hash": snapshot_hash.lower(),
            "evaluation_run_id": f"model-eval-{evaluation_hash[:16]}",
            "evaluation_report_hash": evaluation_hash,
            "model_revision": model_revision,
        },
        "evidence_paths": evidence_paths,
        "forward_edges": {
            edge.edge_id: {
                "edge_id": edge.edge_id,
                "source_id": edge.source_id,
                "relation": edge.relation,
                "target_id": edge.target_id,
            }
            for edge in snapshot.edges
        },
        "answers": {
            row["question_id"]: {
                "status": row["status"],
                "intent": row["intent"],
                "entity_ids": row["entity_ids"],
                "citations": row["citations"],
                "text": row["text"],
                "abstention_reason": row["abstention_reason"],
            }
            for row in rows
        },
    }
    validated = FrozenDemoBundle.model_validate(bundle_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "benchmark_summary.json": _json_bytes(summary),
        "per_query.jsonl": b"".join(_json_bytes(row) for row in rows),
        "demo_bundle.json": _json_bytes(validated.model_dump(mode="json")),
    }
    artifact_hashes: dict[str, str] = {}
    for name, content in artifacts.items():
        path = output_dir / name
        path.write_bytes(content)
        artifact_hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": "1",
        "snapshot_hash": snapshot_hash.lower(),
        "evaluation_report_hash": evaluation_hash,
        "artifacts": artifact_hashes,
    }
    (output_dir / "manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
