from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from knetminer_llm.data.stage import load_staged_snapshot
from knetminer_llm.evaluation.bundle import write_frozen_result_bundle
from knetminer_llm.evaluation.hgt_benchmark import (
    BenchmarkConfig,
    encode_node_labels,
    feature_sha256,
    mean_pool,
    run_hgt_benchmark,
)
from knetminer_llm.evaluation.model_backed import run_model_backed_evaluation
from knetminer_llm.evaluation.questions import EvaluationQuestion
from knetminer_llm.evaluation.runtime import capture_runtime, resolve_device
from knetminer_llm.synthesis.local_llm import (
    LocalQwenConfig,
    LocalQwenRunner,
    QWEN_MODEL_ID,
    QWEN_MODEL_REVISION,
)


MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def model_bundle_config() -> LocalQwenConfig:
    """Return the approved bounded generation configuration for bundle runs."""

    return LocalQwenConfig()


def load_questions(path: Path) -> tuple[EvaluationQuestion, ...]:
    """Load the exact real 30/15 question artifact."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError("question artifact must contain a questions list")
    questions = tuple(
        EvaluationQuestion(
            question_id=record["question_id"],
            answerable=record["answerable"],
            intent_name=record["intent_name"],
            entity_ids=tuple(record["entity_ids"]),
            failure_mode=record.get("failure_mode"),
        )
        for record in records
    )
    if len(questions) != 45 or sum(question.answerable for question in questions) != 30:
        raise ValueError("question artifact must contain exactly 30 answerable and 15 unanswerable rows")
    if len({question.question_id for question in questions}) != 45:
        raise ValueError("question artifact IDs must be unique")
    return questions


def write_json_artifact(path: Path, payload: object) -> str:
    """Write canonical JSON and return the physical file SHA-256."""

    content = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def directory_manifest_hash(path: Path) -> str:
    """Hash every non-cache model file by relative path and content."""

    if not path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {path}")
    digest = hashlib.sha256()
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and ".cache" not in candidate.relative_to(path).parts
    )
    if not files:
        raise ValueError("model directory contains no files")
    for candidate in files:
        digest.update(candidate.relative_to(path).as_posix().encode())
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_model_source(
    *,
    model_id: str,
    revision: str,
    local_path: Path,
    allow_download: bool,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve pinned local files, downloading only with an explicit gate."""

    if local_path.is_dir() and any(local_path.iterdir()):
        return local_path
    if not allow_download:
        raise FileNotFoundError("model files are absent and download approval was not supplied")
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=local_path,
        allow_patterns=allow_patterns,
    )
    return Path(resolved)


def load_minilm_features(snapshot, *, model_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Encode node labels with the pinned MiniLM revision from local files."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
    model.eval()

    def encoder(labels: list[str]) -> torch.Tensor:
        batches = []
        for start in range(0, len(labels), 128):
            encoded = tokenizer(
                labels[start : start + 128],
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded).last_hidden_state
            batches.append(mean_pool(output, encoded["attention_mask"]).cpu())
        return torch.cat(batches) if batches else torch.empty((0, 384))

    return encode_node_labels(snapshot, encoder)


def main(argv: list[str] | None = None) -> int:
    """Run an approved Project 06 evidence gate from the command line."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    snapshot = load_staged_snapshot(args.snapshot, expected_artifact_hash=args.snapshot_sha)
    if args.command == "hgt":
        device = resolve_device(args.device)
        model_path = resolve_model_source(
            model_id=MINILM_MODEL_ID,
            revision=MINILM_MODEL_REVISION,
            local_path=args.minilm_dir,
            allow_download=args.allow_model_download,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "1_Pooling/*", "modules.json"],
        )
        features = load_minilm_features(snapshot, model_path=model_path, device=device)
        feature_hash = feature_sha256(features)
        result = run_hgt_benchmark(
            snapshot,
            snapshot_hash=args.snapshot_sha,
            feature_hash=feature_hash,
            features=features,
            requested_device=args.device,
            config=BenchmarkConfig(max_epochs=args.max_epochs, patience=args.patience),
        )
        result["feature_model"] = {
            "model_id": MINILM_MODEL_ID,
            "revision": MINILM_MODEL_REVISION,
            "directory_hash": directory_manifest_hash(model_path),
        }
        artifact_hash = write_json_artifact(args.output, result)
        print(json.dumps({"status": "passed", "artifact": str(args.output), "sha256": artifact_hash}))
        return 0

    device = resolve_device(args.device)
    model_path = resolve_model_source(
        model_id=QWEN_MODEL_ID,
        revision=QWEN_MODEL_REVISION,
        local_path=args.qwen_dir,
        allow_download=args.allow_model_download,
        allow_patterns=["*.json", "*.txt", "*.model", "*.tiktoken", "*.safetensors", "LICENSE"],
    )
    model_hash = directory_manifest_hash(model_path)
    runner = LocalQwenRunner.from_local_files(
        model_bundle_config(),
        model_path,
        device=device,
    )
    questions = load_questions(args.questions)
    evaluation = run_model_backed_evaluation(snapshot, questions, runner=runner)
    evaluation["summary"]["runtime"] = capture_runtime(
        seed=42,
        requested_device=args.device,
        input_hashes={"model_directory": model_hash, "snapshot": args.snapshot_sha},
    )
    manifest = write_frozen_result_bundle(
        snapshot,
        evaluation,
        snapshot_hash=args.snapshot_sha,
        output_dir=args.output_dir,
        model_revision=QWEN_MODEL_REVISION,
    )
    print(json.dumps({"status": "passed", "output_dir": str(args.output_dir), "manifest": manifest}))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="knetminer-runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--snapshot", type=Path, required=True)
    common.add_argument("--snapshot-sha", required=True)
    common.add_argument("--allow-model-download", action="store_true")

    hgt = subparsers.add_parser("hgt", parents=[common])
    hgt.add_argument("--minilm-dir", type=Path, required=True)
    hgt.add_argument("--output", type=Path, required=True)
    hgt.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    hgt.add_argument("--max-epochs", type=int, default=100)
    hgt.add_argument("--patience", type=int, default=10)

    model = subparsers.add_parser("model-bundle", parents=[common])
    model.add_argument("--questions", type=Path, required=True)
    model.add_argument("--qwen-dir", type=Path, required=True)
    model.add_argument("--output-dir", type=Path, required=True)
    model.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
