from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from knetminer_llm.data.normalize import NormalizedNode, NormalizedSnapshot, normalize_snapshot
from knetminer_llm.data.opentargets import (
    GRAPHQL_ENDPOINT,
    FetchResult,
    OpenTargetsFetcher,
)
from knetminer_llm.contracts import EvidenceEdge, SnapshotManifest
from knetminer_llm.data.validate import validate_snapshot


REAL_SAMPLE_DISEASE_IDS = (
    "MONDO_0005148",  # type 2 diabetes mellitus
    "MONDO_0005147",  # type 1 diabetes mellitus
    "MONDO_0004979",  # asthma
    "MONDO_0004975",  # Alzheimer disease
    "MONDO_0008383",  # rheumatoid arthritis
    "MONDO_0007254",  # breast cancer
    "MONDO_0005233",  # non-small cell lung carcinoma
    "MONDO_0005090",  # schizophrenia
    "MONDO_0005180",  # Parkinson disease
    "MONDO_0005301",  # multiple sclerosis
)

DISEASE_QUERY = """
query DiseaseSample($id: String!, $page: Pagination!) {
  disease(efoId: $id) {
    id
    name
    associatedTargets(page: $page, enableIndirect: false) {
      rows {
        score
        target { id approvedSymbol approvedName }
      }
    }
    drugAndClinicalCandidates {
      rows { id maxClinicalStage drug { id name } }
    }
  }
}
""".strip()

TARGET_QUERY = """
query TargetSample($id: String!) {
  target(ensemblId: $id) {
    id
    approvedSymbol
    approvedName
    drugAndClinicalCandidates {
      rows { id maxClinicalStage drug { id name } }
    }
  }
}
""".strip()


def _edge_id(source_id: str, relation: str, target_id: str) -> str:
    value = f"26.06|{relation}|{source_id}|{target_id}".encode("utf-8")
    return "edge:" + hashlib.sha256(value).hexdigest()


def _combined_hash(values: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _add_node(nodes: dict[str, dict[str, str]], stable_id: str, node_type: str, label: str) -> None:
    if not stable_id or not label:
        raise ValueError("staged records require stable IDs and labels")
    existing = nodes.get(stable_id)
    candidate = {"id": stable_id, "type": node_type, "label": label}
    if existing is not None and existing != candidate:
        raise ValueError(f"conflicting staged node record: {stable_id}")
    nodes[stable_id] = candidate


def _add_edge(
    edges: dict[tuple[str, str, str], dict[str, Any]],
    *,
    source_id: str,
    relation: str,
    target_id: str,
    score: float | None,
    provenance_ids: tuple[str, ...],
) -> None:
    key = (source_id, relation, target_id)
    candidate = {
        "edge_id": _edge_id(source_id, relation, target_id),
        "source_id": source_id,
        "relation": relation,
        "target_id": target_id,
        "score": score,
        "provenance_ids": list(provenance_ids),
    }
    existing = edges.get(key)
    if existing is not None:
        if existing["score"] != score:
            raise ValueError(f"conflicting staged edge score: {key}")
        candidate["provenance_ids"] = sorted(
            {*existing["provenance_ids"], *candidate["provenance_ids"]}
        )
    edges[key] = candidate


def normalize_sample_payloads(
    *,
    disease_results: tuple[FetchResult, ...],
    target_results: tuple[FetchResult, ...],
    retrieved_at: datetime,
) -> NormalizedSnapshot:
    nodes: dict[str, dict[str, str]] = {}
    edges: dict[tuple[str, str, str], dict[str, Any]] = {}

    for result in disease_results:
        disease = result.payload.get("data", {}).get("disease")
        if not isinstance(disease, dict):
            raise ValueError("staged disease response is missing data.disease")
        disease_id = disease["id"]
        _add_node(nodes, disease_id, "disease", disease["name"])
        for row in disease.get("associatedTargets", {}).get("rows", []):
            target = row.get("target")
            if not isinstance(target, dict):
                continue
            target_id = target["id"]
            _add_node(nodes, target_id, "target", target.get("approvedName") or target["approvedSymbol"])
            _add_edge(
                edges,
                source_id=disease_id,
                relation="disease_target",
                target_id=target_id,
                score=row.get("score"),
                provenance_ids=(result.response_hash,),
            )
        for row in disease.get("drugAndClinicalCandidates", {}).get("rows", []):
            drug = row.get("drug")
            if not isinstance(drug, dict):
                continue
            drug_id = drug["id"]
            _add_node(nodes, drug_id, "drug", drug["name"])
            _add_edge(
                edges,
                source_id=disease_id,
                relation="disease_drug",
                target_id=drug_id,
                score=None,
                provenance_ids=tuple(
                    value
                    for value in (result.response_hash, row.get("id"))
                    if isinstance(value, str) and value
                ),
            )

    for result in target_results:
        target = result.payload.get("data", {}).get("target")
        if not isinstance(target, dict):
            raise ValueError("staged target response is missing data.target")
        target_id = target["id"]
        _add_node(nodes, target_id, "target", target.get("approvedName") or target["approvedSymbol"])
        for row in target.get("drugAndClinicalCandidates", {}).get("rows", []):
            drug = row.get("drug")
            if not isinstance(drug, dict):
                continue
            drug_id = drug["id"]
            _add_node(nodes, drug_id, "drug", drug["name"])
            _add_edge(
                edges,
                source_id=target_id,
                relation="target_drug",
                target_id=drug_id,
                score=None,
                provenance_ids=tuple(
                    value
                    for value in (result.response_hash, row.get("id"))
                    if isinstance(value, str) and value
                ),
            )

    request_hashes = tuple(result.request_hash for result in disease_results + target_results)
    response_hashes = tuple(result.response_hash for result in disease_results + target_results)
    return normalize_snapshot(
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
        source_url=GRAPHQL_ENDPOINT,
        retrieved_at=retrieved_at,
        request_hash=_combined_hash(request_hashes),
        response_hash=_combined_hash(response_hashes),
    )


def load_staged_snapshot(path: Path, *, expected_artifact_hash: str) -> NormalizedSnapshot:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_artifact_hash.lower():
        raise ValueError("staged snapshot artifact hash does not match the trusted expected hash")
    payload = json.loads(raw)
    edges = tuple(
        EvidenceEdge.model_validate(
            {**edge, "provenance_ids": tuple(edge["provenance_ids"])}
        )
        for edge in payload["edges"]
    )
    snapshot = NormalizedSnapshot(
        nodes=tuple(NormalizedNode.model_validate(node) for node in payload["nodes"]),
        edges=edges,
        manifest=SnapshotManifest.model_validate(
            {
                **payload["manifest"],
                "retrieved_at": datetime.fromisoformat(
                    payload["manifest"]["retrieved_at"].replace("Z", "+00:00")
                ),
            }
        ),
    )
    return validate_snapshot(snapshot)


async def stage_bounded_snapshot(
    fetcher: OpenTargetsFetcher,
    *,
    disease_ids: tuple[str, ...] = REAL_SAMPLE_DISEASE_IDS,
    page_size: int = 5,
    max_target_queries: int = 20,
    retrieved_at: datetime,
) -> NormalizedSnapshot:
    if not disease_ids or not 1 <= page_size <= 100:
        raise ValueError("bounded staging requires disease IDs and page_size between 1 and 100")
    if max_target_queries < 0:
        raise ValueError("max_target_queries must be non-negative")
    await fetcher.verify_release()
    variables = {"page": {"index": 0, "size": page_size}}
    disease_results = tuple(
        [
            await fetcher.fetch_query(DISEASE_QUERY, {"id": disease_id, **variables})
            for disease_id in disease_ids
        ]
    )
    target_ids: list[str] = []
    for result in disease_results:
        disease = result.payload.get("data", {}).get("disease", {})
        for row in disease.get("associatedTargets", {}).get("rows", []):
            target = row.get("target")
            if isinstance(target, dict) and target["id"] not in target_ids:
                target_ids.append(target["id"])
    target_results = tuple(
        [
            await fetcher.fetch_query(TARGET_QUERY, {"id": target_id})
            for target_id in target_ids[:max_target_queries]
        ]
    )
    return normalize_sample_payloads(
        disease_results=disease_results,
        target_results=target_results,
        retrieved_at=retrieved_at,
    )
