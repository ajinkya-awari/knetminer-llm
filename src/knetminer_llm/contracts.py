from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import AnyUrl, BaseModel, ConfigDict, Field, field_validator, model_validator


SOURCE_RELEASE = "26.06"
SOURCE_LICENCE = "CC0"
ALLOWED_SOURCE_HOSTS = frozenset({"platform.opentargets.org", "api.platform.opentargets.org"})
NODE_TYPES = frozenset({"disease", "target", "drug"})

FORWARD_RELATIONS = {
    "disease_target": ("disease", "target"),
    "disease_drug": ("disease", "drug"),
    "target_drug": ("target", "drug"),
}
REVERSE_RELATIONS = {
    "target_disease": ("target", "disease"),
    "drug_disease": ("drug", "disease"),
    "drug_target": ("drug", "target"),
}
ALL_RELATIONS = FORWARD_RELATIONS | REVERSE_RELATIONS

StableId = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")]
NodeType = Literal["disease", "target", "drug"]
ResolutionMethod = Literal["exact_id", "exact_alias", "semantic"]
IntentName = Literal["shared_targets", "disease_drugs", "target_context"]

_STABLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ResolvedEntity(StrictModel):
    node_type: NodeType
    stable_id: StableId
    label: str = Field(min_length=1)
    resolution_method: ResolutionMethod
    semantic_score: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_margin: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("stable_id")
    @classmethod
    def stable_id_must_be_safe(cls, value: str) -> str:
        if not _STABLE_ID.fullmatch(value):
            raise ValueError("stable_id must be a non-empty stable identifier")
        return value

    @model_validator(mode="after")
    def validate_resolution_scores(self) -> "ResolvedEntity":
        if self.resolution_method == "semantic":
            if self.semantic_score is None or self.semantic_margin is None:
                raise ValueError("semantic resolution requires score and margin")
            if self.semantic_score < 0.75 or self.semantic_margin < 0.05:
                raise ValueError("semantic resolution is below acceptance thresholds")
        elif self.semantic_score is not None or self.semantic_margin is not None:
            raise ValueError("semantic scores are only valid for semantic resolution")
        return self


class Intent(StrictModel):
    name: IntentName
    entity_ids: tuple[StableId, ...] = Field(min_length=1)

    @field_validator("entity_ids")
    @classmethod
    def entity_ids_must_be_safe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not _STABLE_ID.fullmatch(value) for value in values):
            raise ValueError("entity_ids must contain stable identifiers")
        return values

    @model_validator(mode="after")
    def validate_entity_arity(self) -> "Intent":
        expected = {"shared_targets": 2, "disease_drugs": 1, "target_context": 1}[self.name]
        if len(self.entity_ids) != expected:
            raise ValueError(f"{self.name} requires exactly {expected} entities")
        return self


class EvidenceEdge(StrictModel):
    edge_id: str = Field(min_length=1)
    source_id: StableId
    source_type: NodeType
    relation: str
    target_id: StableId
    target_type: NodeType
    release: Literal["26.06"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    provenance_ids: tuple[str, ...] = Field(min_length=1)
    source_url: AnyUrl
    licence: Literal["CC0"]
    observed: Literal[True]
    citable: bool

    @field_validator("source_id", "target_id")
    @classmethod
    def endpoint_ids_must_be_safe(cls, value: str) -> str:
        if not _STABLE_ID.fullmatch(value):
            raise ValueError("endpoint IDs must be stable identifiers")
        return value

    @field_validator("source_url")
    @classmethod
    def source_url_must_be_allow_listed(cls, value: AnyUrl) -> AnyUrl:
        parsed = urlparse(str(value))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("source_url is not on the Open Targets HTTPS allow-list")
        return value

    @model_validator(mode="after")
    def validate_relation_and_citation(self) -> "EvidenceEdge":
        if self.relation not in ALL_RELATIONS:
            raise ValueError("relation is not approved")
        expected_source, expected_target = ALL_RELATIONS[self.relation]
        if (self.source_type, self.target_type) != (expected_source, expected_target):
            raise ValueError("relation endpoint types do not match")
        if self.citable != (self.relation in FORWARD_RELATIONS):
            raise ValueError("only forward relations may be citable")
        return self


class EvidencePath(StrictModel):
    path_id: str = Field(min_length=1)
    edges: tuple[EvidenceEdge, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_observed_chain(self) -> "EvidencePath":
        edge_ids = [edge.edge_id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("path edge IDs must be unique")
        if not any(edge.citable for edge in self.edges):
            raise ValueError("path must contain at least one citable forward edge")
        for previous, current in zip(self.edges, self.edges[1:]):
            if previous.target_id != current.source_id:
                raise ValueError("path edges must form a chained observed path")
        return self


class Claim(StrictModel):
    text: str = Field(min_length=1)
    evidence_path_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("evidence_path_ids")
    @classmethod
    def path_ids_must_be_nonempty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("claim evidence path IDs must be non-empty")
        return values

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("claim text must not be blank")
        return value


class AnswerPayload(StrictModel):
    status: Literal["answered", "abstained"]
    intent: Intent
    claims: tuple[Claim, ...]
    evidence_paths: tuple[EvidencePath, ...]
    citations: tuple[str, ...]
    abstention_reason: str | None = None

    @model_validator(mode="after")
    def validate_claim_evidence(self) -> "AnswerPayload":
        supplied = {path.path_id for path in self.evidence_paths}
        if len(supplied) != len(self.evidence_paths):
            raise ValueError("evidence path IDs must be unique")
        if self.status == "answered":
            if not self.claims:
                raise ValueError("answered payload requires at least one claim")
            if any(not set(claim.evidence_path_ids) <= supplied for claim in self.claims):
                raise ValueError("every claim must cite supplied evidence path IDs")
            if len(self.citations) != len(set(self.citations)):
                raise ValueError("citations must be unique")
            claim_citations = {
                path_id
                for claim in self.claims
                for path_id in claim.evidence_path_ids
            }
            if set(self.citations) != claim_citations or not claim_citations <= supplied:
                raise ValueError("citations must exactly match claim evidence path IDs")
            if self.abstention_reason is not None:
                raise ValueError("answered payload cannot have an abstention reason")
        else:
            if self.claims or self.citations:
                raise ValueError("abstained payload cannot contain claims or citations")
            if not self.abstention_reason or not self.abstention_reason.strip():
                raise ValueError("abstained payload requires a reason")
        return self


class SnapshotManifest(StrictModel):
    source_release: Literal["26.06"]
    licence: Literal["CC0"]
    source_url: AnyUrl
    retrieved_at: datetime
    request_hash: str
    response_hash: str
    node_count: int = Field(ge=0)
    edge_count: int = Field(ge=0)

    @field_validator("source_url")
    @classmethod
    def manifest_url_must_be_allow_listed(cls, value: AnyUrl) -> AnyUrl:
        parsed = urlparse(str(value))
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_SOURCE_HOSTS:
            raise ValueError("source_url is not on the Open Targets HTTPS allow-list")
        return value

    @field_validator("request_hash", "response_hash")
    @classmethod
    def hashes_must_be_sha256(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("hashes must be lowercase or uppercase SHA-256 hex")
        return value.lower()
