from __future__ import annotations

from knetminer_llm.contracts import FORWARD_RELATIONS, NODE_TYPES
from knetminer_llm.data.normalize import NormalizedSnapshot


def validate_snapshot(snapshot: NormalizedSnapshot) -> NormalizedSnapshot:
    if snapshot.manifest.node_count != len(snapshot.nodes):
        raise ValueError("manifest node count does not match normalized nodes")
    if snapshot.manifest.edge_count != len(snapshot.edges):
        raise ValueError("manifest edge count does not match normalized edges")
    if any(node.node_type not in NODE_TYPES for node in snapshot.nodes):
        raise ValueError("normalized nodes contain an unknown node type")
    node_by_id = {node.stable_id: node for node in snapshot.nodes}
    if len(node_by_id) != len(snapshot.nodes):
        raise ValueError("normalized nodes contain duplicate stable IDs")
    if len({edge.edge_id for edge in snapshot.edges}) != len(snapshot.edges):
        raise ValueError("normalized edges contain duplicate edge IDs")
    if any(not edge.observed or not edge.citable for edge in snapshot.edges):
        raise ValueError("normalized snapshot contains non-citable transport edges")
    seen_edge_keys: set[tuple[str, str, str]] = set()
    for edge in snapshot.edges:
        source = node_by_id.get(edge.source_id)
        target = node_by_id.get(edge.target_id)
        if source is None or target is None:
            raise ValueError("edge endpoint is missing from normalized nodes")
        if (source.node_type, target.node_type) != (edge.source_type, edge.target_type):
            raise ValueError("edge endpoint type does not match normalized nodes")
        if edge.relation not in FORWARD_RELATIONS:
            raise ValueError("normalized snapshot accepts only citable forward relations")
        if (source.node_type, target.node_type) != FORWARD_RELATIONS[edge.relation]:
            raise ValueError("relation endpoint types do not match normalized nodes")
        edge_key = (edge.source_id, edge.relation, edge.target_id)
        if edge_key in seen_edge_keys:
            raise ValueError("normalized edges contain duplicate forward edge keys")
        seen_edge_keys.add(edge_key)
    return snapshot
