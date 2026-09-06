from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from knetminer_llm.contracts import (
    FORWARD_RELATIONS,
    EvidenceEdge,
    NodeType,
    SnapshotManifest,
    StableId,
)


class NormalizedNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    stable_id: StableId
    node_type: NodeType
    label: str = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def label_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("node label must not be blank")
        return value


@dataclass(frozen=True)
class NormalizedSnapshot:
    nodes: tuple[NormalizedNode, ...]
    edges: tuple[EvidenceEdge, ...]
    manifest: SnapshotManifest


def normalize_snapshot(
    *,
    nodes: tuple[dict[str, Any], ...],
    edges: tuple[dict[str, Any], ...],
    source_url: str,
    retrieved_at: datetime,
    request_hash: str,
    response_hash: str,
    source_release: str = "26.06",
    licence: str = "CC0",
) -> NormalizedSnapshot:
    if source_release != "26.06":
        raise ValueError("only Open Targets release 26.06 is allowed")
    if licence != "CC0":
        raise ValueError("only CC0 source records are allowed")

    normalized_nodes = tuple(
        NormalizedNode(stable_id=node["id"], node_type=node["type"], label=node["label"])
        for node in nodes
    )
    node_by_id: dict[str, NormalizedNode] = {}
    for node in normalized_nodes:
        if node.node_type not in {"disease", "target", "drug"}:
            raise ValueError("unknown node type")
        if node.stable_id in node_by_id:
            raise ValueError("duplicate stable node ID")
        node_by_id[node.stable_id] = node

    normalized_edges: list[EvidenceEdge] = []
    seen_edge_ids: set[str] = set()
    seen_edge_keys: set[tuple[str, str, str]] = set()
    for raw in edges:
        source_id = raw["source_id"]
        target_id = raw["target_id"]
        relation = raw["relation"]
        if source_id not in node_by_id or target_id not in node_by_id:
            raise ValueError("edge endpoint is missing from normalized nodes")
        if relation not in FORWARD_RELATIONS:
            raise ValueError("normalization accepts only citable forward relations")
        if (node_by_id[source_id].node_type, node_by_id[target_id].node_type) != FORWARD_RELATIONS[relation]:
            raise ValueError("relation endpoint types do not match")
        edge_id = raw["edge_id"]
        edge_key = (source_id, relation, target_id)
        if edge_id in seen_edge_ids or edge_key in seen_edge_keys:
            raise ValueError("duplicate forward edge")
        seen_edge_ids.add(edge_id)
        seen_edge_keys.add(edge_key)
        normalized_edges.append(
            EvidenceEdge(
                edge_id=edge_id,
                source_id=source_id,
                source_type=node_by_id[source_id].node_type,
                relation=relation,
                target_id=target_id,
                target_type=node_by_id[target_id].node_type,
                release="26.06",
                score=raw.get("score"),
                provenance_ids=tuple(raw["provenance_ids"]),
                source_url=source_url,
                licence="CC0",
                observed=True,
                citable=True,
            )
        )

    manifest = SnapshotManifest(
        source_release="26.06",
        licence="CC0",
        source_url=source_url,
        retrieved_at=retrieved_at,
        request_hash=request_hash,
        response_hash=response_hash,
        node_count=len(normalized_nodes),
        edge_count=len(normalized_edges),
    )
    return NormalizedSnapshot(normalized_nodes, tuple(normalized_edges), manifest)
