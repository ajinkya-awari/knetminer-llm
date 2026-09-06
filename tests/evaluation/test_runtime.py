from __future__ import annotations

import math

import pytest

from knetminer_llm.evaluation.runtime import (
    binary_ranking_metrics,
    canonical_sha256,
    capture_runtime,
    filtered_ranking_metrics,
    resolve_device,
)


def test_cuda_request_fails_closed_when_cuda_is_unavailable() -> None:
    if __import__("torch").cuda.is_available():
        pytest.skip("this regression exercises the unavailable-CUDA branch")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_cuda_request_fails_before_work_when_wheel_omits_gpu_architecture(monkeypatch) -> None:
    torch = __import__("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (6, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_70", "sm_80"])

    with pytest.raises(RuntimeError, match="sm_60"):
        resolve_device("cuda")


def test_binary_metrics_match_hand_calculated_ranking_values() -> None:
    metrics = binary_ranking_metrics(
        labels=(1, 0, 1, 0),
        scores=(0.9, 0.8, 0.7, 0.1),
    )
    assert metrics["aucpr"] == pytest.approx(5 / 6)
    assert metrics["roc_auc"] == pytest.approx(3 / 4)


def test_binary_metrics_reject_non_finite_scores_and_single_class_labels() -> None:
    with pytest.raises(ValueError, match="finite"):
        binary_ranking_metrics(labels=(1, 0), scores=(math.nan, 0.0))
    with pytest.raises(ValueError, match="both classes"):
        binary_ranking_metrics(labels=(1, 1), scores=(0.9, 0.8))


def test_filtered_ranking_metrics_match_hand_calculated_ranks() -> None:
    metrics = filtered_ranking_metrics((1, 2, 10))
    assert metrics["filtered_mrr"] == pytest.approx(8 / 15)
    assert metrics["hits_at_5"] == pytest.approx(2 / 3)


def test_runtime_record_is_complete_and_hash_is_canonical() -> None:
    record = capture_runtime(seed=42, requested_device="cpu", input_hashes={"snapshot": "a" * 64})
    assert record["seed"] == 42
    assert record["requested_device"] == "cpu"
    assert record["device_used"] == "cpu"
    assert record["python"]
    assert record["os"]
    assert record["torch"]
    assert record["torch_geometric"]
    assert record["input_hashes"] == {"snapshot": "a" * 64}
    assert canonical_sha256({"b": 1, "a": 2}) == canonical_sha256({"a": 2, "b": 1})
