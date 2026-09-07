from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_does_not_hide_package_model_source() -> None:
    patterns = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "models/" not in patterns
    assert "/models/" in patterns


def test_static_demo_is_a_self_contained_frozen_bundle_workbench() -> None:
    demo = ROOT / "docs" / "demo"
    index = (demo / "index.html").read_text(encoding="utf-8")

    assert (demo / "styles.css").is_file()
    assert (demo / "app.js").is_file()
    assert (demo / "bundle.js").stat().st_size > 1_000
    assert 'content="default-src \'self\'' in index
    assert 'aria-live="polite"' in index
    assert 'id="case-search"' in index
    assert "not for clinical use" in index.lower()


def test_static_demo_has_no_runtime_or_tracking_boundary() -> None:
    script = (ROOT / "docs" / "demo" / "app.js").read_text(encoding="utf-8")
    forbidden = (
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "localStorage",
        "sessionStorage",
        "innerHTML",
        "eval(",
        "analytics",
    )

    assert not any(token in script for token in forbidden)
    assert not re.search(r"https?://", script)
