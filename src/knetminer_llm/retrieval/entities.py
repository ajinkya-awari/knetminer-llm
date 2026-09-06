from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from knetminer_llm.contracts import NODE_TYPES


class ResolutionStatus(str, Enum):
    ACCEPTED = "accepted"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class EntityRecord:
    node_type: str
    stable_id: str
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class SemanticCandidate:
    entity: EntityRecord
    cosine: float


@dataclass(frozen=True)
class ResolutionResult:
    status: ResolutionStatus
    entity: EntityRecord | None
    candidates: tuple[EntityRecord, ...]
    reason: str


def resolve_entity(
    query: str,
    expected_type: str,
    catalog: tuple[EntityRecord, ...],
    *,
    semantic_candidates: tuple[SemanticCandidate, ...] = (),
) -> ResolutionResult:
    _validate_catalog(catalog, expected_type)
    _validate_semantic_candidates(semantic_candidates, catalog)
    text = query.strip()
    exact_id = [entity for entity in catalog if entity.stable_id == text]
    if exact_id:
        entity = exact_id[0]
        if entity.node_type != expected_type:
            return ResolutionResult(
                ResolutionStatus.ABSTAIN,
                None,
                (entity,),
                "stable ID belongs to a different node type",
            )
        return ResolutionResult(ResolutionStatus.ACCEPTED, entity, (entity,), "exact stable ID")

    normalized = _normalize_alias(text)
    alias_matches = [
        entity
        for entity in catalog
        if entity.node_type == expected_type
        and normalized in {_normalize_alias(entity.label), *(_normalize_alias(alias) for alias in entity.aliases)}
    ]
    if len(alias_matches) == 1:
        return ResolutionResult(ResolutionStatus.ACCEPTED, alias_matches[0], tuple(alias_matches), "exact alias")
    if len(alias_matches) > 1:
        return ResolutionResult(
            ResolutionStatus.CLARIFY,
            None,
            tuple(sorted(alias_matches, key=lambda entity: entity.stable_id)),
            "alias matches multiple entities",
        )

    candidates = sorted(
        (candidate for candidate in semantic_candidates if candidate.entity.node_type == expected_type),
        key=lambda candidate: (-candidate.cosine, candidate.entity.stable_id),
    )
    if not candidates:
        return ResolutionResult(ResolutionStatus.ABSTAIN, None, (), "no entity match")
    top = candidates[0]
    margin = top.cosine - candidates[1].cosine if len(candidates) > 1 else 0.0
    ordered_entities = tuple(candidate.entity for candidate in candidates)
    if top.cosine >= 0.75 and margin >= 0.05:
        return ResolutionResult(ResolutionStatus.ACCEPTED, top.entity, ordered_entities, "semantic threshold")
    return ResolutionResult(ResolutionStatus.CLARIFY, None, ordered_entities, "semantic score or margin is uncertain")


def _normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold().strip())


def _validate_catalog(catalog: tuple[EntityRecord, ...], expected_type: str) -> None:
    if expected_type not in NODE_TYPES:
        raise ValueError("expected entity type is not approved")
    seen_ids: set[str] = set()
    for entity in catalog:
        if entity.node_type not in NODE_TYPES:
            raise ValueError("catalog contains an unknown node type")
        if entity.stable_id in seen_ids:
            raise ValueError("catalog contains duplicate stable IDs")
        seen_ids.add(entity.stable_id)


def _validate_semantic_candidates(
    semantic_candidates: tuple[SemanticCandidate, ...],
    catalog: tuple[EntityRecord, ...],
) -> None:
    catalog_members = set(catalog)
    seen_candidate_ids: set[str] = set()
    for candidate in semantic_candidates:
        if candidate.entity not in catalog_members:
            raise ValueError("semantic candidate is not a catalog member")
        if candidate.entity.stable_id in seen_candidate_ids:
            raise ValueError("semantic candidates contain duplicate stable IDs")
        seen_candidate_ids.add(candidate.entity.stable_id)
        if not math.isfinite(candidate.cosine) or not 0.0 <= candidate.cosine <= 1.0:
            raise ValueError("semantic score must be finite and between zero and one")
