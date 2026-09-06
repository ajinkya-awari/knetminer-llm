from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from knetminer_llm.contracts import (
    ALLOWED_SOURCE_HOSTS,
    FORWARD_RELATIONS,
    IntentName,
    StableId,
)
from knetminer_llm.synthesis.local_llm import QWEN_MODEL_REVISION
from knetminer_llm.synthesis.validate import SUPPORTED_CLAIM_TEXT


class BundleError(RuntimeError):
    """Raised when the frozen demo bundle is missing or invalid."""


class BundleProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    source_release: Literal["26.06"]
    licence: Literal["CC0"]
    source_url: AnyUrl
    request_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    response_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    snapshot_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    artifact_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    evaluation_run_id: str = Field(min_length=1)
    evaluation_report_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    model_revision: Literal[QWEN_MODEL_REVISION, "deterministic-only"]

    @field_validator("source_url")
    @classmethod
    def source_url_must_be_allow_listed(cls, value: AnyUrl) -> AnyUrl:
        parsed = urlparse(str(value))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("source_url is not on the Open Targets HTTPS allow-list")
        return value

    @model_validator(mode="after")
    def bind_snapshot_to_artifact(self) -> "BundleProvenance":
        if self.snapshot_hash != self.artifact_hash:
            raise ValueError("snapshot hash must match the staged artifact hash")
        return self


class FrozenAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: Literal["answered", "abstained"]
    intent: IntentName
    entity_ids: list[StableId] = Field(min_length=1)
    citations: list[str] = Field(default_factory=list)
    text: str | None = None
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> "FrozenAnswer":
        expected_arity = {"shared_targets": 2, "disease_drugs": 1, "target_context": 1}[self.intent]
        if len(self.entity_ids) != expected_arity:
            raise ValueError("frozen answer entity IDs do not match intent arity")
        if len(self.citations) != len(set(self.citations)):
            raise ValueError("citations must be unique")
        if any(not citation.strip() for citation in self.citations):
            raise ValueError("citations must be non-empty")
        if self.status == "answered":
            if not self.citations:
                raise ValueError("answered records require citations")
            if not self.text or not self.text.strip():
                raise ValueError("answered records require text")
            if self.abstention_reason is not None:
                raise ValueError("answered records cannot include an abstention reason")
            if self.text != SUPPORTED_CLAIM_TEXT:
                raise ValueError("answered text is not supported by the release contract")
        else:
            if self.citations or self.text is not None:
                raise ValueError("abstained records cannot include citations or answer text")
            if not self.abstention_reason or not self.abstention_reason.strip():
                raise ValueError("abstained records require an abstention reason")
        return self


class FrozenEvidencePath(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path_id: str = Field(min_length=1)
    node_ids: list[StableId] = Field(min_length=2, max_length=4)
    relations: list[str] = Field(min_length=1, max_length=3)
    citation_edge_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> "FrozenEvidencePath":
        if len(self.relations) != len(self.node_ids) - 1:
            raise ValueError("evidence-path relation count must match its node chain")
        if len(self.citation_edge_ids) != len(self.relations):
            raise ValueError("each evidence-path segment requires one forward citation edge")
        if len(self.citation_edge_ids) != len(set(self.citation_edge_ids)):
            raise ValueError("evidence-path citation edge IDs must be unique")
        if any(not edge_id.strip() or edge_id.startswith("reverse:") for edge_id in self.citation_edge_ids):
            raise ValueError("evidence paths require non-empty forward citation edge IDs")
        return self


class FrozenForwardEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    edge_id: str = Field(min_length=1)
    source_id: StableId
    relation: str
    target_id: StableId

    @field_validator("relation")
    @classmethod
    def relation_must_be_forward(cls, value: str) -> str:
        if value not in FORWARD_RELATIONS:
            raise ValueError("frozen snapshot inventory may contain only forward relations")
        return value


class FrozenDemoBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal["1"]
    provenance: BundleProvenance
    evidence_paths: dict[str, FrozenEvidencePath]
    forward_edges: dict[str, FrozenForwardEdge]
    answers: dict[str, FrozenAnswer]

    @model_validator(mode="after")
    def validate_answer_evidence(self) -> "FrozenDemoBundle":
        if any(path_id != path.path_id for path_id, path in self.evidence_paths.items()):
            raise ValueError("evidence path keys must match their path IDs")
        supplied = set(self.evidence_paths)
        if any(not set(answer.citations) <= supplied for answer in self.answers.values()):
            raise ValueError("answer citations must reference bundled evidence paths")
        if not self.forward_edges:
            raise ValueError("frozen snapshot forward-edge inventory must not be empty")
        if any(edge_id != edge.edge_id for edge_id, edge in self.forward_edges.items()):
            raise ValueError("snapshot edge keys must match their edge IDs")
        for path in self.evidence_paths.values():
            for index, edge_id in enumerate(path.citation_edge_ids):
                edge = self.forward_edges.get(edge_id)
                if edge is None:
                    raise ValueError("evidence paths must cite edges in the frozen snapshot")
                if not _edge_matches_path_segment(edge, path, index):
                    raise ValueError("cited snapshot edge does not match evidence-path topology")
        for answer in self.answers.values():
            if answer.status == "answered" and any(
                not _path_matches_answer(answer, self.evidence_paths[path_id])
                for path_id in answer.citations
            ):
                raise ValueError("answer evidence path does not match its intent and entities")
        return self


def _edge_matches_path_segment(
    edge: FrozenForwardEdge,
    path: FrozenEvidencePath,
    index: int,
) -> bool:
    relation = path.relations[index]
    source_id = path.node_ids[index]
    target_id = path.node_ids[index + 1]
    reverse_to_forward = {
        "target_disease": "disease_target",
        "drug_disease": "disease_drug",
        "drug_target": "target_drug",
    }
    if relation in FORWARD_RELATIONS:
        return (
            edge.source_id == source_id
            and edge.relation == relation
            and edge.target_id == target_id
        )
    forward_relation = reverse_to_forward.get(relation)
    return (
        forward_relation is not None
        and edge.source_id == target_id
        and edge.relation == forward_relation
        and edge.target_id == source_id
    )


def _path_matches_answer(answer: FrozenAnswer, path: FrozenEvidencePath) -> bool:
    relations = tuple(path.relations)
    if answer.intent == "shared_targets":
        return (
            path.node_ids[0] == answer.entity_ids[0]
            and path.node_ids[-1] == answer.entity_ids[1]
            and relations == ("disease_target", "target_disease")
        )
    if answer.intent == "disease_drugs":
        return path.node_ids[0] == answer.entity_ids[0] and relations in {
            ("disease_drug",),
            ("disease_target", "target_drug"),
        }
    return path.node_ids[0] == answer.entity_ids[0] and relations in {
        ("target_disease",),
        ("target_drug",),
        ("target_disease", "disease_drug"),
    }


def load_frozen_bundle(path: Path, *, expected_snapshot_hash: str) -> FrozenDemoBundle:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundle = FrozenDemoBundle.model_validate(payload)
        if bundle.provenance.snapshot_hash != expected_snapshot_hash:
            raise BundleError("frozen demo bundle snapshot hash does not match expected staged artifact")
        return bundle
    except (OSError, json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise BundleError(f"frozen demo bundle is missing or invalid: {path}") from exc


def lookup_answer(bundle: FrozenDemoBundle, question_id: str) -> dict[str, Any] | None:
    answer = bundle.answers.get(question_id)
    return answer.model_dump() if answer is not None else None


def static_catalogue(bundle: FrozenDemoBundle) -> tuple[str, ...]:
    return tuple(sorted(bundle.answers))


def render_static_html(bundle: FrozenDemoBundle) -> str:
    """Render a deterministic, network-free catalogue for static fallback hosting."""

    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="utf-8">',
        "  <title>KnetMiner Evidence Benchmark</title>",
        "</head>",
        "<body>",
        "  <main>",
        "    <h1>KnetMiner Evidence Benchmark</h1>",
        f"    <p>Open Targets {_escape(bundle.provenance.source_release)} · "
        f"{_escape(bundle.provenance.licence)} · frozen local catalogue</p>",
        "    <p>This is evidence retrieval, not clinical advice.</p>",
    ]
    for question_id in static_catalogue(bundle):
        answer = bundle.answers[question_id]
        detail = answer.text or answer.abstention_reason or "No additional text provided."
        citations = ", ".join(answer.citations) or "None"
        lines.extend(
            [
                f'    <article data-question-id="{_escape(question_id)}">',
                f"      <h2>{_escape(question_id)}</h2>",
                f"      <p>Status: {_escape(answer.status)}</p>",
                f"      <p>{_escape(detail)}</p>",
                f"      <p>Citations: {_escape(citations)}</p>",
                "    </article>",
            ]
        )
    lines.extend(["  </main>", "</body>", "</html>", ""])
    return "\n".join(lines)


def _escape(value: object) -> str:
    return escape(str(value), quote=True)
