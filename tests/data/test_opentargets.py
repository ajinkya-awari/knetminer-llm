from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from knetminer_llm.data.normalize import normalize_snapshot
from knetminer_llm.data.opentargets import (
    GraphQLError,
    OpenTargetsFetcher,
    ReleaseMismatch,
    ReleaseNotVerified,
)


ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
RELEASE_METADATA = {"data": {"meta": {"dataVersion": {"year": "26", "month": "06"}}}}


def response(status: int, payload: dict, headers: dict[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=payload, headers=headers or {})


def transport_for(responses: list[httpx.Response]) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.platform.opentargets.org"
        if not responses:
            raise AssertionError("fixture transport exhausted")
        return responses.pop(0)

    return httpx.MockTransport(handler)


def run(coro):
    return asyncio.run(coro)


def test_metadata_release_must_be_verified_before_graphql_query() -> None:
    with pytest.raises(TypeError):
        OpenTargetsFetcher(expected_release="25.06")

    fetcher = OpenTargetsFetcher(transport=transport_for([]))
    with pytest.raises(ReleaseNotVerified):
        run(fetcher.fetch_query("query { data }", {}))
    run(fetcher.aclose())

    mismatch = OpenTargetsFetcher(
        transport=transport_for([response(200, {"data": {"release": "25.06"}})])
    )
    with pytest.raises(ReleaseMismatch):
        run(mismatch.verify_release())
    run(mismatch.aclose())


def test_metadata_rejects_legacy_release_only_shape() -> None:
    fetcher = OpenTargetsFetcher(
        transport=transport_for([response(200, {"data": {"release": "26.06"}})])
    )

    with pytest.raises(ReleaseMismatch):
        run(fetcher.verify_release())
    run(fetcher.aclose())


def test_metadata_verification_uses_graphql_meta_data_version() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response(200, {"data": {"meta": {"dataVersion": {"year": "26", "month": "06"}}}})

    fetcher = OpenTargetsFetcher(transport=httpx.MockTransport(handler))
    run(fetcher.verify_release())
    assert requests[0].method == "POST"
    assert requests[0].url.path == "/api/v4/graphql"
    assert "dataVersion" in requests[0].content.decode("utf-8")
    run(fetcher.aclose())


def test_graphql_errors_on_http_200_fail_without_retry() -> None:
    fetcher = OpenTargetsFetcher(
        transport=transport_for(
            [
                response(200, RELEASE_METADATA),
                response(200, {"errors": [{"message": "schema failure"}]}),
            ]
        )
    )

    run(fetcher.verify_release())
    with pytest.raises(GraphQLError, match="schema failure"):
        run(fetcher.fetch_query("query { data }", {}))
    assert fetcher.attempt_count == 2
    run(fetcher.aclose())


def test_retry_after_is_honored_and_retry_is_bounded() -> None:
    sleeps: list[float] = []
    fetcher = OpenTargetsFetcher(
        transport=transport_for(
            [
                response(200, RELEASE_METADATA),
                response(429, {"error": "busy"}, {"Retry-After": "2"}),
                response(200, {"data": {"ok": True}}),
            ]
        ),
        sleep=lambda seconds: sleeps.append(seconds),
        jitter=lambda: 0.25,
    )

    run(fetcher.verify_release())
    result = run(fetcher.fetch_query("query { data }", {}))
    assert result.payload == {"data": {"ok": True}}
    assert sleeps == [2.25]
    assert fetcher.attempt_count == 3
    run(fetcher.aclose())


def test_successful_response_is_content_addressed_and_cached() -> None:
    fetcher = OpenTargetsFetcher(
        transport=transport_for(
            [
                response(200, RELEASE_METADATA),
                response(200, {"data": {"ok": True}}),
            ]
        )
    )

    run(fetcher.verify_release())
    first = run(fetcher.fetch_query("query { data }", {"id": "EFO_0001"}))
    second = run(fetcher.fetch_query("query { data }", {"id": "EFO_0001"}))
    assert first.request_hash == second.request_hash
    assert first.response_hash == second.response_hash
    assert second.cache_hit is True
    assert fetcher.attempt_count == 2
    run(fetcher.aclose())


def test_cached_payload_cannot_be_mutated_through_a_prior_result() -> None:
    fetcher = OpenTargetsFetcher(
        transport=transport_for(
            [
                response(200, RELEASE_METADATA),
                response(200, {"data": {"nested": {"value": "original"}}}),
            ]
        )
    )
    run(fetcher.verify_release())
    first = run(fetcher.fetch_query("query { data }", {}))
    first.payload["data"]["nested"]["value"] = "tampered"

    cached = run(fetcher.fetch_query("query { data }", {}))

    assert cached.payload["data"]["nested"]["value"] == "original"
    run(fetcher.aclose())


def test_normalization_preserves_ids_provenance_and_manifest() -> None:
    snapshot = normalize_snapshot(
        nodes=(
            {"id": "EFO_0001", "type": "disease", "label": "Disease"},
            {"id": "ENSG0001", "type": "target", "label": "Target"},
            {"id": "CHEMBL1", "type": "drug", "label": "Drug"},
        ),
        edges=(
            {
                "edge_id": "edge-1",
                "source_id": "EFO_0001",
                "relation": "disease_target",
                "target_id": "ENSG0001",
                "score": 0.9,
                "provenance_ids": ["assoc-1"],
            },
            {
                "edge_id": "edge-2",
                "source_id": "EFO_0001",
                "relation": "disease_drug",
                "target_id": "CHEMBL1",
                "score": 0.7,
                "provenance_ids": ["assoc-2"],
            },
        ),
        source_url="https://platform.opentargets.org/",
        retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        request_hash="a" * 64,
        response_hash="b" * 64,
    )

    assert tuple(node.stable_id for node in snapshot.nodes) == ("EFO_0001", "ENSG0001", "CHEMBL1")
    assert tuple(edge.edge_id for edge in snapshot.edges) == ("edge-1", "edge-2")
    assert snapshot.edges[0].provenance_ids == ("assoc-1",)
    assert snapshot.manifest.edge_count == 2


@pytest.mark.parametrize(
    "mutator",
    [
        lambda nodes, edges: (nodes + ({"id": "EFO_0001", "type": "disease", "label": "Duplicate"},), edges),
        lambda nodes, edges: (nodes, edges + ({"edge_id": "edge-missing", "source_id": "EFO_9999", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.5, "provenance_ids": ["p"]},)),
        lambda nodes, edges: (nodes, edges + (edges[0],)),
    ],
)
def test_normalization_rejects_duplicate_ids_missing_endpoints_and_duplicate_edges(mutator) -> None:
    nodes = (
        {"id": "EFO_0001", "type": "disease", "label": "Disease"},
        {"id": "ENSG0001", "type": "target", "label": "Target"},
    )
    edges = (
        {"edge_id": "edge-1", "source_id": "EFO_0001", "relation": "disease_target", "target_id": "ENSG0001", "score": 0.9, "provenance_ids": ["assoc-1"]},
    )
    changed_nodes, changed_edges = mutator(nodes, edges)
    with pytest.raises(ValueError):
        normalize_snapshot(
            nodes=changed_nodes,
            edges=changed_edges,
            source_url="https://platform.opentargets.org/",
            retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            request_hash="a" * 64,
            response_hash="b" * 64,
        )


def test_normalization_rejects_restricted_licence() -> None:
    with pytest.raises(ValueError, match="CC0"):
        normalize_snapshot(
            nodes=({"id": "EFO_0001", "type": "disease", "label": "Disease"},),
            edges=(),
            source_url="https://platform.opentargets.org/",
            retrieved_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
            request_hash="a" * 64,
            response_hash="b" * 64,
            licence="CC-BY",
        )
