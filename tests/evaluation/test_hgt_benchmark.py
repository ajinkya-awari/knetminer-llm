from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch

from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.evaluation.hgt_benchmark import (
    BenchmarkConfig,
    encode_node_labels,
    feature_sha256,
    filter_message_passing_edges,
    mean_pool,
    run_hgt_benchmark,
    run_seed_benchmark,
)
from knetminer_llm.graph.build import build_graph_views
from knetminer_llm.graph.split import audit_no_leakage, split_forward_edges


def benchmark_snapshot():
    nodes = []
    edges = []
    for index in range(6):
        disease = f"EFO_{index:04d}"
        target = f"ENSG{index:011d}"
        drug = f"CHEMBL{index + 1}"
        nodes.extend(
            [
                {"id": disease, "type": "disease", "label": disease},
                {"id": target, "type": "target", "label": target},
                {"id": drug, "type": "drug", "label": drug},
            ]
        )
        edges.extend(
            [
                {"edge_id": f"dt-{index}", "source_id": disease, "relation": "disease_target", "target_id": target, "score": 0.8, "provenance_ids": [f"p-{index}"]},
                {"edge_id": f"dd-{index}", "source_id": disease, "relation": "disease_drug", "target_id": drug, "score": 0.7, "provenance_ids": [f"pd-{index}"]},
            ]
        )
    return normalize_snapshot(
        nodes=tuple(nodes),
        edges=tuple(edges),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )


def test_benchmark_config_enforces_exact_seeds_and_training_bounds() -> None:
    BenchmarkConfig()
    with pytest.raises(ValueError, match="42, 43, and 44"):
        BenchmarkConfig(seeds=(42,))
    with pytest.raises(ValueError, match="100"):
        BenchmarkConfig(max_epochs=101)
    with pytest.raises(ValueError, match="patience"):
        BenchmarkConfig(patience=11)


def test_message_passing_filter_removes_held_out_edges_and_reverse_twins() -> None:
    snapshot = benchmark_snapshot()
    views = build_graph_views(snapshot)
    split = split_forward_edges(snapshot.edges, seed=42)
    audit_no_leakage(split)

    edge_index = filter_message_passing_edges(
        views,
        allowed_edge_ids=frozenset(split.message_passing_edge_ids),
        device=torch.device("cpu"),
    )

    retained_count = sum(value.shape[1] for value in edge_index.values())
    assert retained_count == len(split.message_passing_edge_ids)
    assert not ({*split.validation_edge_ids, *split.test_edge_ids} & set(split.message_passing_edge_ids))


def test_one_seed_benchmark_returns_finite_complete_metrics_without_creating_evidence() -> None:
    snapshot = benchmark_snapshot()
    views = build_graph_views(snapshot)
    features = {
        node_type: torch.arange(views.pyg[node_type].num_nodes * 384, dtype=torch.float32).reshape(-1, 384) / 1000
        for node_type in views.pyg.node_types
    }

    result = run_seed_benchmark(
        snapshot,
        seed=42,
        device=torch.device("cpu"),
        features=features,
        max_epochs=1,
        patience=1,
    )

    assert result["seed"] == 42
    assert result["device_used"] == "cpu"
    assert result["epochs_completed"] == 1
    assert result["observed_test_edges"] == 1
    assert set(result["metrics"]) == {"aucpr", "roc_auc", "filtered_mrr", "hits_at_5"}
    assert all(0.0 <= value <= 1.0 for value in result["metrics"].values())
    assert "predicted_edges" not in result


def test_label_encoder_preserves_graph_order_and_requires_384_dimensions() -> None:
    snapshot = benchmark_snapshot()
    seen = []

    def encoder(labels):
        seen.append(tuple(labels))
        return torch.ones((len(labels), 384))

    features = encode_node_labels(snapshot, encoder)

    assert set(features) == {"disease", "target", "drug"}
    assert seen[0] == tuple(f"EFO_{index:04d}" for index in range(6))
    assert features["target"].shape == (6, 384)


def test_three_seed_benchmark_records_runtime_and_aggregate_metrics() -> None:
    snapshot = benchmark_snapshot()
    views = build_graph_views(snapshot)
    features = {
        node_type: torch.arange(views.pyg[node_type].num_nodes * 384, dtype=torch.float32).reshape(-1, 384) / 1000
        for node_type in views.pyg.node_types
    }

    result = run_hgt_benchmark(
        snapshot,
        snapshot_hash="a" * 64,
        feature_hash="b" * 64,
        features=features,
        requested_device="cpu",
        config=BenchmarkConfig(max_epochs=1, patience=1),
    )

    assert [run["seed"] for run in result["runs"]] == [42, 43, 44]
    assert set(result["aggregate"]) == {"aucpr", "roc_auc", "filtered_mrr", "hits_at_5"}
    assert all(set(values) == {"mean", "standard_deviation"} for values in result["aggregate"].values())
    assert result["runtime"]["requested_device"] == "cpu"
    assert result["input_hashes"] == {"features": "b" * 64, "snapshot": "a" * 64}


def test_masked_mean_pooling_and_feature_hash_are_deterministic() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]], [[2.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[1, 1], [1, 0]])

    pooled = mean_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
    features = {"disease": pooled, "target": pooled + 1, "drug": pooled + 2}
    assert feature_sha256(features) == feature_sha256(dict(reversed(tuple(features.items()))))
