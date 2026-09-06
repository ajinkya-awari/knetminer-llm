from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Callable

import networkx as nx

from knetminer_llm.contracts import Intent
from knetminer_llm.graph.build import GraphViews


MAX_DEPTH = 3
MAX_EXPANSIONS = 2_000
MAX_SECONDS = 2.0
MAX_CANDIDATES = 100
MAX_OUTPUTS = 5


@dataclass(frozen=True)
class PathCandidate:
    path_id: str
    edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    citation_edge_ids: tuple[str, ...]
    symbolic_score: float
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalResult:
    paths: tuple[PathCandidate, ...]
    expansions: int
    truncated: bool


def retrieve_paths(
    views: GraphViews,
    intent: Intent,
    *,
    max_depth: int = MAX_DEPTH,
    max_expansions: int = MAX_EXPANSIONS,
    max_seconds: float = MAX_SECONDS,
    max_candidates: int = MAX_CANDIDATES,
    max_outputs: int = MAX_OUTPUTS,
    clock: Callable[[], float] = time.monotonic,
) -> RetrievalResult:
    if max_depth < 1 or max_depth > MAX_DEPTH:
        raise ValueError("path depth must be between one and three edges")
    if not 1 <= max_expansions <= MAX_EXPANSIONS:
        raise ValueError("expansion cap must be between one and 2,000")
    if not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("time cap must be greater than zero and at most two seconds")
    if not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("candidate cap must be between one and 100")
    if not 1 <= max_outputs <= MAX_OUTPUTS:
        raise ValueError("output cap must be between one and five")
    start_id = intent.entity_ids[0]
    if start_id not in views.networkx:
        raise ValueError("unknown start entity")
    destination_id = intent.entity_ids[1] if intent.name == "shared_targets" else None
    queue = deque([(start_id, (start_id,), (), frozenset({start_id}))])
    candidates: list[PathCandidate] = []
    seen_paths: set[str] = set()
    expansions = 0
    started = clock()
    truncated = False

    while queue:
        if expansions >= max_expansions or clock() - started >= max_seconds:
            truncated = True
            break
        current, node_ids, edge_ids, visited = queue.popleft()
        outgoing = sorted(
            views.networkx.out_edges(current, keys=True, data=True),
            key=lambda item: (item[1], item[3]["relation"], item[2]),
        )
        for _, target, edge_id, attributes in outgoing:
            if expansions >= max_expansions or clock() - started >= max_seconds:
                truncated = True
                break
            expansions += 1
            if target in visited:
                continue
            next_edges = edge_ids + (edge_id,)
            next_nodes = node_ids + (target,)
            next_visited = visited | {target}
            if _matches_intent(
                views.networkx,
                intent.name,
                target,
                destination_id,
                next_nodes,
                next_edges,
            ):
                candidate = _candidate(next_nodes, next_edges, views.networkx)
                if candidate.path_id not in seen_paths:
                    candidates.append(candidate)
                    seen_paths.add(candidate.path_id)
                    if len(candidates) >= max_candidates:
                        truncated = bool(queue)
                        queue.clear()
                        break
            if len(next_edges) < max_depth:
                queue.append((target, next_nodes, next_edges, next_visited))

    candidates.sort(key=lambda path: (-path.symbolic_score, path.path_id))
    return RetrievalResult(tuple(candidates[:max_outputs]), expansions, truncated or bool(queue))


def rerank_paths(
    paths: tuple[PathCandidate, ...],
    scores: dict[str, float],
) -> tuple[PathCandidate, ...]:
    path_ids = {path.path_id for path in paths}
    if set(scores) != path_ids:
        raise ValueError("HGT reranking scores contain unknown or missing path IDs")
    if any(not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores.values()):
        raise ValueError("HGT reranking scores must be finite numbers")
    return tuple(
        sorted(
            (replace(path, rerank_score=scores[path.path_id]) for path in paths),
            key=lambda path: (-path.rerank_score, path.path_id),
        )
    )


def _matches_intent(
    graph: nx.MultiDiGraph,
    intent_name: str,
    target: str,
    destination_id: str | None,
    node_ids: tuple[str, ...],
    edge_ids: tuple[str, ...],
) -> bool:
    node_type = graph.nodes[target]["node_type"]
    relations = tuple(
        graph[node_ids[index]][node_ids[index + 1]][edge_id]["relation"]
        for index, edge_id in enumerate(edge_ids)
    )
    if intent_name == "shared_targets":
        return target == destination_id and relations == ("disease_target", "target_disease")
    if intent_name == "disease_drugs":
        return node_type == "drug" and relations in {
            ("disease_drug",),
            ("disease_target", "target_drug"),
        }
    if intent_name == "target_context":
        return node_type in {"disease", "drug"} and relations in {
            ("target_disease",),
            ("target_drug",),
            ("target_disease", "disease_drug"),
        }
    raise ValueError("unsupported intent")


def _candidate(
    node_ids: tuple[str, ...],
    edge_ids: tuple[str, ...],
    graph: nx.MultiDiGraph,
) -> PathCandidate:
    scores: list[float] = []
    citation_ids: list[str] = []
    for index, edge_id in enumerate(edge_ids):
        attributes = graph[node_ids[index]][node_ids[index + 1]][edge_id]
        score = attributes["score"]
        scores.append(1.0 if score is None else score)
        forward_id = edge_id if attributes["citable"] else attributes["forward_edge_id"]
        if forward_id not in citation_ids:
            citation_ids.append(forward_id)
    symbolic_score = math.prod(scores) ** (1.0 / len(scores))
    path_id = "path:" + "|".join(edge_ids)
    return PathCandidate(path_id, edge_ids, node_ids, tuple(citation_ids), symbolic_score)
