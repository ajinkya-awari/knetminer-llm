from __future__ import annotations

import copy
import hashlib
import statistics
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as functional

from knetminer_llm.data.normalize import NormalizedSnapshot
from knetminer_llm.evaluation.runtime import (
    binary_ranking_metrics,
    capture_runtime,
    filtered_ranking_metrics,
    resolve_device,
)
from knetminer_llm.graph.build import GraphViews, build_graph_views
from knetminer_llm.graph.split import (
    NegativeEdge,
    audit_no_leakage,
    sample_negative_edges,
    split_forward_edges,
)
from knetminer_llm.models.hgt import HGTConfig, HGTReranker


@dataclass(frozen=True)
class BenchmarkConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.seeds != (42, 43, 44):
            raise ValueError("Project 06 requires exactly seeds 42, 43, and 44")
        if not 1 <= self.max_epochs <= 100:
            raise ValueError("maximum epochs must be between 1 and 100")
        if not 1 <= self.patience <= 10:
            raise ValueError("patience must be between 1 and 10")
        if self.learning_rate != 1e-3 or self.weight_decay != 1e-4:
            raise ValueError("Project 06 requires AdamW lr=1e-3 and weight_decay=1e-4")


def encode_node_labels(
    snapshot: NormalizedSnapshot,
    encoder: Callable[[list[str]], Tensor],
) -> dict[str, Tensor]:
    """Encode labels in the same stable order used by the graph views."""

    result: dict[str, Tensor] = {}
    for node_type in ("disease", "target", "drug"):
        labels = [node.label for node in snapshot.nodes if node.node_type == node_type]
        values = encoder(labels)
        if not isinstance(values, Tensor) or values.shape != (len(labels), 384):
            raise ValueError("node label encoder must return one 384-dimensional row per label")
        if not torch.isfinite(values).all():
            raise ValueError("node label embeddings must be finite")
        result[node_type] = values.detach().cpu().to(torch.float32)
    return result


def mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool token embeddings while excluding padded positions."""

    expanded = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    denominator = expanded.sum(dim=1).clamp_min(1.0)
    return (last_hidden_state * expanded).sum(dim=1) / denominator


def feature_sha256(features: dict[str, Tensor]) -> str:
    """Hash ordered tensor names, shapes, dtypes, and bytes."""

    digest = hashlib.sha256()
    for node_type in sorted(features):
        value = features[node_type].detach().cpu().contiguous()
        digest.update(node_type.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def run_hgt_benchmark(
    snapshot: NormalizedSnapshot,
    *,
    snapshot_hash: str,
    feature_hash: str,
    features: dict[str, Tensor],
    requested_device: str,
    config: BenchmarkConfig,
) -> dict[str, object]:
    """Run the exact three-seed benchmark and aggregate every required metric."""

    device = resolve_device(requested_device)
    input_hashes = {"features": feature_hash.lower(), "snapshot": snapshot_hash.lower()}
    runs = [
        run_seed_benchmark(
            snapshot,
            seed=seed,
            device=device,
            features=features,
            max_epochs=config.max_epochs,
            patience=config.patience,
        )
        for seed in config.seeds
    ]
    aggregate = {}
    for metric_name in ("aucpr", "roc_auc", "filtered_mrr", "hits_at_5"):
        values = [float(run["metrics"][metric_name]) for run in runs]
        aggregate[metric_name] = {
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.pstdev(values),
        }
    runtime = capture_runtime(seed=42, requested_device=requested_device, input_hashes=input_hashes)
    runtime["seeds"] = list(config.seeds)
    return {
        "benchmark_type": "bounded_observed_disease_target_hgt",
        "feature_identity": "pinned_minilm_node_label_embeddings",
        "input_hashes": input_hashes,
        "runtime": runtime,
        "parameters": {
            "max_epochs": config.max_epochs,
            "patience": config.patience,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
        },
        "runs": runs,
        "aggregate": aggregate,
    }


def filter_message_passing_edges(
    views: GraphViews,
    *,
    allowed_edge_ids: frozenset[str],
    device: torch.device,
) -> dict[tuple[str, str, str], Tensor]:
    """Return only explicitly retained forward edges and transport twins."""

    result: dict[tuple[str, str, str], Tensor] = {}
    retained: set[str] = set()
    for edge_type in views.pyg.edge_types:
        storage = views.pyg[edge_type]
        indices = [index for index, edge_id in enumerate(storage.edge_ids) if edge_id in allowed_edge_ids]
        if indices:
            result[edge_type] = storage.edge_index[:, indices].to(device)
            retained.update(storage.edge_ids[index] for index in indices)
    if retained != set(allowed_edge_ids):
        raise ValueError("message-passing edge inventory does not match the leakage-safe split")
    return result


def run_seed_benchmark(
    snapshot: NormalizedSnapshot,
    *,
    seed: int,
    device: torch.device,
    features: dict[str, Tensor],
    max_epochs: int,
    patience: int,
) -> dict[str, object]:
    """Train and evaluate one leakage-safe HGT seed on known snapshot edges."""

    if seed not in {42, 43, 44}:
        raise ValueError("seed must be one of 42, 43, and 44")
    if not 1 <= max_epochs <= 100 or not 1 <= patience <= 10:
        raise ValueError("training bounds exceed the Project 06 contract")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    views = build_graph_views(snapshot)
    _validate_features(views, features)
    split = split_forward_edges(snapshot.edges, seed=seed)
    audit_no_leakage(split)
    edge_index_dict = filter_message_passing_edges(
        views,
        allowed_edge_ids=frozenset(split.message_passing_edge_ids),
        device=device,
    )
    x_dict = {
        node_type: functional.normalize(value.to(device), p=2, dim=1)
        for node_type, value in features.items()
    }
    model = HGTReranker(
        views.pyg.metadata(),
        HGTConfig(input_dims={node_type: 384 for node_type in ("disease", "target", "drug")}),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    disease_target_edges = tuple(edge for edge in snapshot.edges if edge.relation == "disease_target")
    by_id = {edge.edge_id: edge for edge in disease_target_edges}
    all_negatives = sample_negative_edges(
        snapshot.nodes,
        disease_target_edges,
        count=len(disease_target_edges),
        seed=seed,
    )
    train_count = len(split.train_edge_ids)
    validation_count = len(split.validation_edge_ids)
    negative_groups = {
        "train": all_negatives[:train_count],
        "validation": all_negatives[train_count : train_count + validation_count],
        "test": all_negatives[train_count + validation_count :],
    }
    positive_groups = {
        "train": tuple(by_id[edge_id] for edge_id in split.train_edge_ids),
        "validation": tuple(by_id[edge_id] for edge_id in split.validation_edge_ids),
        "test": tuple(by_id[edge_id] for edge_id in split.test_edge_ids),
    }
    node_indices = {
        node_type: {node_id: index for index, node_id in enumerate(views.pyg[node_type].node_ids)}
        for node_type in views.pyg.node_types
    }

    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(x_dict, edge_index_dict)
        loss = _classification_loss(
            model,
            embeddings,
            positive_groups["train"],
            negative_groups["train"],
            node_indices,
        )
        loss.backward()
        optimizer.step()
        epochs_completed = epoch + 1

        model.eval()
        with torch.no_grad():
            embeddings = model(x_dict, edge_index_dict)
            validation_positives = positive_groups["validation"] or positive_groups["train"]
            validation_negatives = negative_groups["validation"] or negative_groups["train"]
            validation_loss = float(
                _classification_loss(
                    model,
                    embeddings,
                    validation_positives,
                    validation_negatives,
                    node_indices,
                ).item()
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("HGT training did not produce a finite validation state")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embeddings = model(x_dict, edge_index_dict)
        positive_scores = _scores(model, embeddings, positive_groups["test"], node_indices)
        negative_scores = _scores(model, embeddings, negative_groups["test"], node_indices)
        binary = binary_ranking_metrics(
            labels=(1,) * len(positive_scores) + (0,) * len(negative_scores),
            scores=tuple(positive_scores + negative_scores),
        )
        ranks = _filtered_ranks(
            model,
            embeddings,
            positive_groups["test"],
            disease_target_edges,
            node_indices,
        )
    return {
        "seed": seed,
        "device_used": device.type,
        "epochs_completed": epochs_completed,
        "observed_train_edges": len(positive_groups["train"]),
        "observed_validation_edges": len(positive_groups["validation"]),
        "observed_test_edges": len(positive_groups["test"]),
        "metrics": {**binary, **filtered_ranking_metrics(ranks)},
    }


def _validate_features(views: GraphViews, features: dict[str, Tensor]) -> None:
    if set(features) != {"disease", "target", "drug"}:
        raise ValueError("features must cover exactly the three node types")
    for node_type, values in features.items():
        if values.shape != (views.pyg[node_type].num_nodes, 384):
            raise ValueError("features must match node counts and have width 384")
        if not torch.isfinite(values).all():
            raise ValueError("features must be finite")


def _pairs(records, node_indices: dict[str, dict[str, int]]) -> tuple[tuple[str, int, str, int], ...]:
    return tuple(
        (
            "disease",
            node_indices["disease"][record.source_id],
            "target",
            node_indices["target"][record.target_id],
        )
        for record in records
    )


def _scores(model, embeddings, records, node_indices) -> list[float]:
    return model.score_observed_pairs(embeddings, _pairs(records, node_indices)).detach().cpu().tolist()


def _classification_loss(model, embeddings, positives, negatives, node_indices) -> Tensor:
    positive_scores = model.score_observed_pairs(embeddings, _pairs(positives, node_indices))
    negative_scores = model.score_observed_pairs(embeddings, _pairs(negatives, node_indices))
    logits = torch.cat((positive_scores, negative_scores))
    labels = torch.cat((torch.ones_like(positive_scores), torch.zeros_like(negative_scores)))
    return functional.binary_cross_entropy_with_logits(logits, labels)


def _filtered_ranks(model, embeddings, positives, all_positives, node_indices) -> tuple[int, ...]:
    known = {}
    for edge in all_positives:
        known.setdefault(edge.source_id, set()).add(edge.target_id)
    target_ids = tuple(node_indices["target"])
    ranks = []
    for edge in positives:
        candidates = tuple(
            target_id
            for target_id in target_ids
            if target_id == edge.target_id or target_id not in known[edge.source_id]
        )
        records = tuple(
            NegativeEdge(edge.source_id, "disease_target", target_id) for target_id in candidates
        )
        scores = _scores(model, embeddings, records, node_indices)
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], candidates[index]))
        ranks.append(order.index(candidates.index(edge.target_id)) + 1)
    return tuple(ranks)
