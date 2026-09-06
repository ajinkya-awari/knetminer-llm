from __future__ import annotations

import json

import pytest

from knetminer_llm.app import (
    BundleError,
    load_frozen_bundle,
    lookup_answer,
    render_static_html,
    static_catalogue,
)


def bundle_payload() -> dict:
    return {
        "schema_version": "1",
        "provenance": {
            "source_release": "26.06",
            "licence": "CC0",
            "snapshot_hash": "a" * 64,
            "source_url": "https://api.platform.opentargets.org/api/v4/graphql",
            "request_hash": "b" * 64,
            "response_hash": "c" * 64,
            "artifact_hash": "a" * 64,
            "evaluation_run_id": "eval-run-2026-08-26",
            "evaluation_report_hash": "d" * 64,
            "model_revision": "deterministic-only",
        },
        "evidence_paths": {
            "path-1": {
                "path_id": "path-1",
                "node_ids": ["EFO_0001", "CHEMBL1"],
                "relations": ["disease_drug"],
                "citation_edge_ids": ["edge-1"],
            },
        },
        "forward_edges": {
            "edge-1": {
                "edge_id": "edge-1",
                "source_id": "EFO_0001",
                "relation": "disease_drug",
                "target_id": "CHEMBL1",
            },
        },
        "answers": {
            "answerable-00": {"status": "answered", "intent": "disease_drugs", "entity_ids": ["EFO_0001"], "citations": ["path-1"], "text": "An observed evidence path is available."},
            "unanswerable-00": {"status": "abstained", "intent": "shared_targets", "entity_ids": ["EFO_0001", "EFO_0002"], "citations": [], "abstention_reason": "No observed path."},
        },
    }


def test_app_loads_frozen_bundle_and_static_catalogue(tmp_path) -> None:
    path = tmp_path / "demo_bundle.json"
    path.write_text(json.dumps(bundle_payload()), encoding="utf-8")

    bundle = load_frozen_bundle(path, expected_snapshot_hash="a" * 64)
    assert lookup_answer(bundle, "answerable-00")["status"] == "answered"
    assert static_catalogue(bundle) == ("answerable-00", "unanswerable-00")

    bound = load_frozen_bundle(path, expected_snapshot_hash="a" * 64)
    assert bound.provenance.artifact_hash == "a" * 64


@pytest.mark.parametrize("content", [None, "not-json", json.dumps({"schema_version": "bad"})])
def test_app_fails_closed_for_missing_or_corrupt_bundle(tmp_path, content) -> None:
    path = tmp_path / "demo_bundle.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(BundleError):
        load_frozen_bundle(path, expected_snapshot_hash="a" * 64)


def test_app_does_not_invent_missing_answer() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    bundle = FrozenDemoBundle.model_validate(bundle_payload())
    assert lookup_answer(bundle, "missing") is None


def test_static_renderer_is_deterministic_and_escapes_bundle_content() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["answers"]["<script>alert(1)</script>"] = {
        "status": "answered",
        "intent": "disease_drugs",
        "entity_ids": ["EFO_0001"],
        "citations": ["path-1"],
        "text": "An observed evidence path is available.",
    }
    bundle = FrozenDemoBundle.model_validate(payload)

    rendered = render_static_html(bundle)

    assert rendered == render_static_html(bundle)
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered
    assert "<script>alert(1)</script>" not in rendered
    assert "Open Targets 26.06" in rendered
    assert "An observed evidence path is available." in rendered


def test_app_rejects_answer_status_citation_mismatches() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    answered_without_citations = bundle_payload()
    answered_without_citations["answers"]["answerable-00"] = {
        "status": "answered",
        "intent": "disease_drugs",
        "entity_ids": ["EFO_0001"],
        "citations": [],
        "text": "Unsupported answer.",
    }
    with pytest.raises(ValueError, match="citations"):
        FrozenDemoBundle.model_validate(answered_without_citations)

    abstained_with_citations = bundle_payload()
    abstained_with_citations["answers"]["unanswerable-00"] = {
        "status": "abstained",
        "intent": "shared_targets",
        "entity_ids": ["EFO_0001", "EFO_0002"],
        "citations": ["path-1"],
        "abstention_reason": "No observed path.",
    }
    with pytest.raises(ValueError, match="abstained"):
        FrozenDemoBundle.model_validate(abstained_with_citations)


def test_app_rejects_unbound_or_mismatched_bundle_provenance(tmp_path) -> None:
    from knetminer_llm.app import FrozenDemoBundle

    missing_endpoint = bundle_payload()
    del missing_endpoint["provenance"]["source_url"]
    with pytest.raises(ValueError):
        FrozenDemoBundle.model_validate(missing_endpoint)

    mismatch = bundle_payload()
    mismatch["provenance"]["artifact_hash"] = "e" * 64
    with pytest.raises(ValueError, match="snapshot"):
        FrozenDemoBundle.model_validate(mismatch)

    path = tmp_path / "demo_bundle.json"
    path.write_text(json.dumps(bundle_payload()), encoding="utf-8")
    with pytest.raises(BundleError, match="snapshot hash"):
        load_frozen_bundle(path, expected_snapshot_hash="f" * 64)


@pytest.mark.parametrize(
    "answer",
    [
        {"status": "unknown", "citations": []},
        {"status": "answered", "citations": [1]},
        {"status": "abstained", "citations": [], "extra": True},
    ],
)
def test_app_rejects_untyped_answer_records(answer) -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["answers"]["bad"] = answer
    with pytest.raises(ValueError):
        FrozenDemoBundle.model_validate(payload)


def test_app_rejects_answer_citation_missing_from_bundle_evidence() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["answers"]["answerable-00"]["citations"] = ["unknown-path"]

    with pytest.raises(ValueError, match="evidence"):
        FrozenDemoBundle.model_validate(payload)


def test_app_rejects_evidence_edge_missing_from_snapshot_inventory() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["evidence_paths"]["path-1"]["citation_edge_ids"] = ["edge-unknown"]

    with pytest.raises(ValueError, match="snapshot"):
        FrozenDemoBundle.model_validate(payload)


def test_app_rejects_evidence_edge_that_does_not_match_path_topology() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["forward_edges"]["edge-1"]["source_id"] = "EFO_9999"

    with pytest.raises(ValueError, match="topology"):
        FrozenDemoBundle.model_validate(payload)


def test_app_rejects_path_that_does_not_match_answer_intent() -> None:
    from knetminer_llm.app import FrozenDemoBundle

    payload = bundle_payload()
    payload["evidence_paths"]["path-1"]["node_ids"] = ["EFO_0002", "CHEMBL1"]
    payload["forward_edges"]["edge-1"]["source_id"] = "EFO_0002"

    with pytest.raises(ValueError, match="intent"):
        FrozenDemoBundle.model_validate(payload)
