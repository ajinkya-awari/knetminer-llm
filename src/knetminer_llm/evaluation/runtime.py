from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from typing import Literal

import torch
import torch_geometric


DeviceName = Literal["cpu", "cuda"]


def resolve_device(requested_device: DeviceName) -> torch.device:
    """Resolve an explicit device request without silently falling back."""

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; CPU fallback is forbidden")
    if requested_device not in {"cpu", "cuda"}:
        raise ValueError("requested device must be 'cpu' or 'cuda'")
    if requested_device == "cuda":
        major, minor = torch.cuda.get_device_capability()
        compiled_arches = tuple(arch for arch in torch.cuda.get_arch_list() if arch.startswith("sm_"))
        supports_major = any(int(arch.removeprefix("sm_")) // 10 == major for arch in compiled_arches)
        if compiled_arches and not supports_major:
            raise RuntimeError(
                f"CUDA device sm_{major}{minor} is unsupported by this PyTorch wheel; "
                f"compiled architectures: {', '.join(compiled_arches)}"
            )
    return torch.device(requested_device)


def binary_ranking_metrics(
    *, labels: tuple[int, ...], scores: tuple[float, ...]
) -> dict[str, float]:
    """Calculate average precision and pairwise ROC-AUC without extra dependencies."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("labels and scores must be non-empty and equal length")
    if any(label not in {0, 1} for label in labels):
        raise ValueError("labels must be binary")
    if any(not math.isfinite(score) for score in scores):
        raise ValueError("scores must be finite")
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("metrics require both classes")

    ordered = sorted(zip(scores, labels), key=lambda item: -item[0])
    hits = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            hits += 1
            precision_sum += hits / rank
    aucpr = precision_sum / len(positives)

    pair_credit = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    roc_auc = pair_credit / (len(positives) * len(negatives))
    return {"aucpr": aucpr, "roc_auc": roc_auc}


def filtered_ranking_metrics(ranks: tuple[int, ...]) -> dict[str, float]:
    """Calculate filtered MRR and Hits@5 from one-indexed positive ranks."""

    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("filtered ranks must be positive integers")
    return {
        "filtered_mrr": sum(1.0 / rank for rank in ranks) / len(ranks),
        "hits_at_5": sum(rank <= 5 for rank in ranks) / len(ranks),
    }


def canonical_sha256(payload: object) -> str:
    """Hash a JSON-compatible value using canonical UTF-8 serialization."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def capture_runtime(
    *, seed: int, requested_device: DeviceName, input_hashes: dict[str, str]
) -> dict[str, object]:
    """Capture the runtime and accelerator identity for one evidence-producing run."""

    device = resolve_device(requested_device)
    if any(len(value) != 64 or any(char not in "0123456789abcdefABCDEF" for char in value) for value in input_hashes.values()):
        raise ValueError("input hashes must be SHA-256 hex values")
    gpu_name: str | None = None
    gpu_memory_bytes: int | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = properties.name
        gpu_memory_bytes = properties.total_memory
    return {
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "requested_device": requested_device,
        "device_used": device.type,
        "gpu_name": gpu_name,
        "gpu_memory_bytes": gpu_memory_bytes,
        "seed": seed,
        "input_hashes": dict(sorted(input_hashes.items())),
    }
