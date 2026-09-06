from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import torch
from torch_geometric.data import HeteroData

from knetminer_llm.contracts import FORWARD_RELATIONS, REVERSE_RELATIONS
from knetminer_llm.data.normalize import NormalizedSnapshot
from knetminer_llm.data.validate import validate_snapshot


REVERSE_FOR_FORWARD = {
    "disease_target": "target_disease",
    "disease_drug": "drug_disease",
    "target_drug": "drug_target",
}


@dataclass(frozen=True)
class GraphViews:
    networkx: nx.MultiDiGraph
    pyg: HeteroData
    forward_edge_ids: tuple[str, ...]
    edge_relations: dict[str, str]


def build_graph_views(snapshot: NormalizedSnapshot) -> GraphViews:
    validate_snapshot(snapshot)
    graph = nx.MultiDiGraph()
    pyg = HeteroData()
    node_indices = {node_type: {} for node_type in ("disease", "target", "drug")}

    for node_type in node_indices:
        node_ids = [node.stable_id for node in snapshot.nodes if node.node_type == node_type]
        node_indices[node_type] = {stable_id: index for index, stable_id in enumerate(node_ids)}
        graph_nodes = [(stable_id, {"node_type": node_type}) for stable_id in node_ids]
        graph.add_nodes_from(graph_nodes)
        pyg[node_type].node_ids = node_ids
        pyg[node_type].num_nodes = len(node_ids)

    forward_ids: list[str] = []
    edge_relations: dict[str, str] = {}
    grouped: dict[tuple[str, str, str], list[tuple[int, int, str, bool, str | None]]] = {}
    for edge in snapshot.edges:
        forward_ids.append(edge.edge_id)
        edge_relations[edge.edge_id] = edge.relation
        _add_networkx_edge(
            graph,
            edge.edge_id,
            edge.source_id,
            edge.target_id,
            edge.relation,
            edge.edge_id,
            True,
            edge.score,
        )
        _add_pyg_edge(
            grouped,
            edge.source_type,
            edge.relation,
            edge.target_type,
            node_indices[edge.source_type][edge.source_id],
            node_indices[edge.target_type][edge.target_id],
            edge.edge_id,
            True,
            None,
        )

        reverse_relation = REVERSE_FOR_FORWARD[edge.relation]
        reverse_edge_id = f"reverse:{edge.edge_id}"
        _add_networkx_edge(
            graph,
            reverse_edge_id,
            edge.target_id,
            edge.source_id,
            reverse_relation,
            edge.edge_id,
            False,
            edge.score,
        )
        _add_pyg_edge(
            grouped,
            edge.target_type,
            reverse_relation,
            edge.source_type,
            node_indices[edge.target_type][edge.target_id],
            node_indices[edge.source_type][edge.source_id],
            reverse_edge_id,
            False,
            edge.edge_id,
        )

    for edge_type, values in grouped.items():
        source, relation, target = edge_type
        pyg[edge_type].edge_index = torch.tensor(
            [[value[0] for value in values], [value[1] for value in values]],
            dtype=torch.long,
        )
        pyg[edge_type].edge_ids = [value[2] for value in values]
        pyg[edge_type].citable = [value[3] for value in values]
        pyg[edge_type].forward_edge_ids = [value[4] for value in values]

    views = GraphViews(graph, pyg, tuple(forward_ids), edge_relations)
    validate_graph_views(views)
    return views


def validate_graph_views(views: GraphViews) -> None:
    if set(views.pyg.node_types) != {"disease", "target", "drug"}:
        raise ValueError("PyG graph must contain exactly the three node types")

    networkx_forward_ids: set[str] = set()
    for _, _, edge_id, attributes in views.networkx.edges(keys=True, data=True):
        relation = attributes["relation"]
        if relation in FORWARD_RELATIONS:
            if not attributes["citable"] or attributes["transport_only"]:
                raise ValueError("forward edges must be citable and not transport-only")
            networkx_forward_ids.add(edge_id)
        elif relation in REVERSE_RELATIONS:
            if attributes["citable"] or not attributes["transport_only"]:
                raise ValueError("reverse edges must be transport-only and non-citable")
        else:
            raise ValueError("graph contains unknown relation")
    if set(views.forward_edge_ids) != networkx_forward_ids:
        raise ValueError("forward edge IDs differ between graph views")
    if any(edge_id not in views.edge_relations for edge_id in views.forward_edge_ids):
        raise ValueError("forward edge IDs lack relation metadata")

    for node_type in ("disease", "target", "drug"):
        networkx_ids = {
            node_id
            for node_id, attributes in views.networkx.nodes(data=True)
            if attributes["node_type"] == node_type
        }
        if set(views.pyg[node_type].node_ids) != networkx_ids:
            raise ValueError("PyG and NetworkX node IDs differ")

    pyg_forward_ids: set[str] = set()
    for source_type, relation, target_type in views.pyg.edge_types:
        storage = views.pyg[(source_type, relation, target_type)]
        if not (
            len(storage.edge_ids)
            == len(storage.citable)
            == len(storage.forward_edge_ids)
            == storage.edge_index.shape[1]
        ):
            raise ValueError("PyG edge metadata lengths differ")
        if relation in FORWARD_RELATIONS:
            if not all(storage.citable) or any(storage.forward_edge_ids):
                raise ValueError("PyG forward edge metadata is invalid")
            pyg_forward_ids.update(storage.edge_ids)
        elif relation in REVERSE_RELATIONS:
            if any(storage.citable) or any(value is None for value in storage.forward_edge_ids):
                raise ValueError("PyG reverse edge metadata is invalid")
        else:
            raise ValueError("PyG graph contains unknown relation")
    if pyg_forward_ids != networkx_forward_ids:
        raise ValueError("forward edge IDs differ between PyG and NetworkX")

    if views.pyg.validate(raise_on_error=True) is not True:
        raise ValueError("PyG graph validation failed")


def _add_networkx_edge(
    graph: nx.MultiDiGraph,
    edge_id: str,
    source_id: str,
    target_id: str,
    relation: str,
    forward_edge_id: str,
    citable: bool,
    score: float | None,
) -> None:
    graph.add_edge(
        source_id,
        target_id,
        key=edge_id,
        edge_id=edge_id,
        relation=relation,
        forward_edge_id=forward_edge_id,
        citable=citable,
        transport_only=not citable,
        score=score,
    )


def _add_pyg_edge(
    grouped: dict[tuple[str, str, str], list[tuple[int, int, str, bool, str | None]]],
    source_type: str,
    relation: str,
    target_type: str,
    source_index: int,
    target_index: int,
    edge_id: str,
    citable: bool,
    forward_edge_id: str | None,
) -> None:
    grouped.setdefault((source_type, relation, target_type), []).append(
        (source_index, target_index, edge_id, citable, forward_edge_id)
    )
