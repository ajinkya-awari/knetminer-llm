from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import httpx


GRAPHQL_ENDPOINT = "https://api.platform.opentargets.org/api/v4/graphql"
METADATA_ENDPOINT = GRAPHQL_ENDPOINT
METADATA_QUERY = "query Metadata { meta { dataVersion { year month } } }"
EXPECTED_RELEASE = "26.06"
RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


class ReleaseNotVerified(RuntimeError):
    """Raised when a data query is attempted before metadata verification."""


class ReleaseMismatch(RuntimeError):
    """Raised when metadata does not match the pinned source release."""


class GraphQLError(RuntimeError):
    """Raised for an API-level GraphQL error, including HTTP 200 responses."""


@dataclass(frozen=True)
class FetchResult:
    payload: dict[str, Any]
    request_hash: str
    response_hash: str
    cache_hit: bool
    attempts: int


SleepFn = Callable[[float], Awaitable[None] | None]
JitterFn = Callable[[], float]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _request_hash(method: str, url: str, body: dict[str, Any] | None) -> str:
    encoded = json.dumps(
        {"method": method, "url": url, "body": body},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


class OpenTargetsFetcher:
    """Bounded, metadata-first client with an injected offline-test transport."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        max_attempts: int = 3,
        sleep: SleepFn | None = None,
        jitter: JitterFn | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.sleep = sleep or _async_sleep
        self.jitter = jitter or (lambda: 0.0)
        self.client = httpx.AsyncClient(
            transport=transport,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=4),
            timeout=10.0,
        )
        self._cache: dict[str, FetchResult] = {}
        self._release_verified = False
        self.attempt_count = 0

    async def aclose(self) -> None:
        await self.client.aclose()

    async def verify_release(self) -> None:
        result = await self._request_json(
            "POST",
            METADATA_ENDPOINT,
            {"query": METADATA_QUERY, "variables": {}},
        )
        data = result.payload.get("data", result.payload)
        release = _release_from_metadata(data)
        if release != EXPECTED_RELEASE:
            raise ReleaseMismatch(f"expected release {EXPECTED_RELEASE}, received {release}")
        self._release_verified = True

    async def fetch_query(self, query: str, variables: dict[str, Any]) -> FetchResult:
        if not self._release_verified:
            raise ReleaseNotVerified("verify_release must succeed before a GraphQL query")
        return await self._request_json(
            "POST",
            GRAPHQL_ENDPOINT,
            {"query": query, "variables": variables},
        )

    async def _request_json(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
    ) -> FetchResult:
        key = _request_hash(method, url, body)
        cached = self._cache.get(key)
        if cached is not None:
            return FetchResult(
                payload=deepcopy(cached.payload),
                request_hash=cached.request_hash,
                response_hash=cached.response_hash,
                cache_hit=True,
                attempts=0,
            )

        for attempt in range(1, self.max_attempts + 1):
            self.attempt_count += 1
            response = await self.client.request(method, url, json=body)
            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == self.max_attempts:
                    response.raise_for_status()
                delay = _retry_after(response) + self.jitter()
                await _maybe_await(self.sleep(max(0.0, delay)))
                continue
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Open Targets response must be a JSON object")
            if payload.get("errors"):
                messages = "; ".join(str(item.get("message", item)) for item in payload["errors"])
                raise GraphQLError(messages)
            cached_result = FetchResult(
                payload=deepcopy(payload),
                request_hash=key,
                response_hash=_sha256_bytes(response.content),
                cache_hit=False,
                attempts=attempt,
            )
            self._cache[key] = cached_result
            return FetchResult(
                payload=deepcopy(cached_result.payload),
                request_hash=cached_result.request_hash,
                response_hash=cached_result.response_hash,
                cache_hit=False,
                attempts=attempt,
            )
        raise AssertionError("retry loop exhausted unexpectedly")


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


async def _maybe_await(value: Any) -> None:
    if inspect.isawaitable(value):
        await value


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is not None:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 0.0


def _release_from_metadata(data: Any) -> str | None:
    if not isinstance(data, dict):
        return None
    meta = data.get("meta")
    if isinstance(meta, dict):
        version = meta.get("dataVersion")
        if isinstance(version, dict):
            year = version.get("year")
            month = version.get("month")
            if isinstance(year, str) and isinstance(month, str):
                return f"{year}.{month}"
    return None
