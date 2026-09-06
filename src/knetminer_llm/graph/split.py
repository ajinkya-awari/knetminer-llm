from __future__ import annotations

import random
from dataclasses import dataclass

from knetminer_llm.contracts import FORWARD_RELATIONS, EvidenceEdge
from knetminer_llm.data.normalize import NormalizedNode


@dataclass(frozen=True)
class SplitResult:
    train_edge_ids: tuple[str, ...]
    validation_edge_ids: tuple[str, ...]
    test_edge_ids: tuple[str, ...]
    context_edge_ids: tuple[str, ...]
    message_passing_edge_ids: tuple[str, ...]


@dataclass(frozen=True)
class NegativeEdge:
    source_id: str
    relation: str
    target_id: str


def split_forward_edges(edges: tuple[EvidenceEdge, ...], *, seed: int = 42) -> SplitResult:
    if not edges:
        raise ValueError("at least one forward edge is required")
    if any(edge.relation not in FORWARD_RELATIONS or not edge.citable for edge in edges):
        raise ValueError("split accepts only citable forward edges")
    all_edge_ids = [edge.edge_id for edge in edges]
    if len(set(all_edge_ids)) != len(all_edge_ids):
        raise ValueError("forward edge IDs must be unique")
    edge_ids = [edge.edge_id for edge in edges if edge.relation == "disease_target"]
    if not edge_ids:
        raise ValueError("at least one disease_target edge is required")
    context = tuple(edge.edge_id for edge in edges if edge.relation != "disease_target")
    rng = random.Random(seed)
    rng.shuffle(edge_ids)
    train_count = max(1, int(len(edge_ids) * 0.70))
    validation_count = max(1, int(len(edge_ids) * 0.15)) if len(edge_ids) >= 3 else 0
    if train_count + validation_count >= len(edge_ids):
        validation_count = max(0, len(edge_ids) - train_count - 1)
    train = tuple(edge_ids[:train_count])
    validation_end = train_count + validation_count
    validation = tuple(edge_ids[train_count:validation_end])
    test = tuple(edge_ids[validation_end:])
    retained = train + context
    message_passing = retained + tuple(f"reverse:{edge_id}" for edge_id in retained)
    return SplitResult(train, validation, test, context, message_passing)


def audit_no_leakage(split: SplitResult) -> None:
    held_out = set(split.validation_edge_ids) | set(split.test_edge_ids)
    message_ids = set(split.message_passing_edge_ids)
    if message_ids & held_out:
        raise ValueError("message passing contains held-out edges")
    held_out_reverse = {f"reverse:{edge_id}" for edge_id in held_out}
    if message_ids & held_out_reverse:
        raise ValueError("message passing contains a held-out reverse twin")
    if set(split.train_edge_ids) & held_out:
        raise ValueError("train and held-out edge IDs overlap")
    retained = set(split.train_edge_ids) | set(split.context_edge_ids)
    expected = retained | {f"reverse:{edge_id}" for edge_id in retained}
    if message_ids != expected:
        raise ValueError("message passing contains unexpected or missing edge IDs")


def sample_negative_edges(
    nodes: tuple[NormalizedNode, ...],
    positive_edges: tuple[EvidenceEdge, ...],
    *,
    count: int,
    seed: int,
) -> tuple[NegativeEdge, ...]:
    if count < 1:
        raise ValueError("negative sample count must be positive")
    nodes_by_type: dict[str, tuple[NormalizedNode, ...]] = {
        node_type: tuple(node for node in nodes if node.node_type == node_type)
        for node_type in ("disease", "target", "drug")
    }
    if any(edge.relation != "disease_target" for edge in positive_edges):
        raise ValueError("negative sampling accepts only disease_target positives")
    positives = {(edge.source_id, edge.relation, edge.target_id) for edge in positive_edges}
    candidates: list[NegativeEdge] = []
    for source in nodes_by_type["disease"]:
        for target in nodes_by_type["target"]:
            candidate = NegativeEdge(source.stable_id, "disease_target", target.stable_id)
            if (candidate.source_id, candidate.relation, candidate.target_id) not in positives:
                candidates.append(candidate)
    if count > len(candidates):
        raise ValueError("requested negative samples exceed available type-correct non-edges")
    random.Random(seed).shuffle(candidates)
    return tuple(candidates[:count])
