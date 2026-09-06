from __future__ import annotations

from dataclasses import replace

import pytest

from knetminer_llm.data.normalize import NormalizedNode
from knetminer_llm.graph.split import (
    audit_no_leakage,
    sample_negative_edges,
    split_forward_edges,
)
from knetminer_llm.contracts import EvidenceEdge


def edges() -> tuple[EvidenceEdge, ...]:
    return tuple(
        EvidenceEdge(
            edge_id=f"edge-{index}",
            source_id=f"EFO_{index:04d}",
            source_type="disease",
            relation="disease_target",
            target_id=f"ENSG{index:04d}",
            target_type="target",
            release="26.06",
            score=0.5,
            provenance_ids=(f"p-{index}",),
            source_url="https://platform.opentargets.org/",
            licence="CC0",
            observed=True,
            citable=True,
        )
        for index in range(10)
    )


def test_split_is_deterministic_and_uses_70_15_15_forward_partition() -> None:
    first = split_forward_edges(edges(), seed=42)
    second = split_forward_edges(edges(), seed=42)

    assert first == second
    assert len(first.train_edge_ids) == 7
    assert len(first.validation_edge_ids) == 1
    assert len(first.test_edge_ids) == 2
    assert set(first.message_passing_edge_ids) == {
        *first.train_edge_ids,
        *(f"reverse:{edge_id}" for edge_id in first.train_edge_ids),
    }


def test_split_uses_only_disease_target_labels_and_preserves_context_edges() -> None:
    context = EvidenceEdge(
        edge_id="context-edge",
        source_id="EFO_0000",
        source_type="disease",
        relation="disease_drug",
        target_id="CHEMBL1",
        target_type="drug",
        release="26.06",
        score=0.5,
        provenance_ids=("context",),
        source_url="https://platform.opentargets.org/",
        licence="CC0",
        observed=True,
        citable=True,
    )

    split = split_forward_edges(edges() + (context,), seed=42)

    supervised = set(split.train_edge_ids + split.validation_edge_ids + split.test_edge_ids)
    assert supervised == {edge.edge_id for edge in edges()}
    assert "context-edge" in split.message_passing_edge_ids
    assert "reverse:context-edge" in split.message_passing_edge_ids


def test_leakage_audit_rejects_held_out_or_reverse_message_edges() -> None:
    split = split_forward_edges(edges(), seed=42)
    audit_no_leakage(split)

    broken = replace(
        split,
        message_passing_edge_ids=split.message_passing_edge_ids + (split.test_edge_ids[0],),
    )
    with pytest.raises(ValueError, match="held-out"):
        audit_no_leakage(broken)

    broken_reverse = replace(
        split,
        message_passing_edge_ids=split.message_passing_edge_ids + ("reverse:unknown",),
    )
    with pytest.raises(ValueError, match="unexpected"):
        audit_no_leakage(broken_reverse)

    held_out_reverse = replace(
        split,
        message_passing_edge_ids=split.message_passing_edge_ids + (f"reverse:{split.test_edge_ids[0]}",),
    )
    with pytest.raises(ValueError, match="reverse twin"):
        audit_no_leakage(held_out_reverse)


def test_negative_sampling_excludes_all_known_positive_typed_edges() -> None:
    nodes = tuple(
        NormalizedNode(stable_id=stable_id, node_type=node_type, label=stable_id)
        for stable_id, node_type in (
            ("EFO_0001", "disease"),
            ("EFO_0002", "disease"),
            ("ENSG0001", "target"),
            ("ENSG0002", "target"),
        )
    )
    negatives = sample_negative_edges(nodes, edges()[:2], count=2, seed=42)
    positives = {(edge.source_id, edge.relation, edge.target_id) for edge in edges()[:2]}
    assert len(negatives) == 2
    assert all((edge.source_id, edge.relation, edge.target_id) not in positives for edge in negatives)
    assert all(edge.relation == "disease_target" for edge in negatives)


def test_negative_sampling_rejects_impossible_request() -> None:
    nodes = (
        NormalizedNode(stable_id="EFO_0001", node_type="disease", label="Disease"),
        NormalizedNode(stable_id="ENSG0001", node_type="target", label="Target"),
    )
    with pytest.raises(ValueError, match="negative"):
        sample_negative_edges(nodes, edges()[:1], count=100, seed=42)
