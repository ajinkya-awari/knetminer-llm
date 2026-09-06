from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "kaggle_run_06-knetminer-llm.ipynb"
KAGGLE_REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements-kaggle.txt"


def test_kaggle_notebook_is_public_source_compatible_and_syntax_valid() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )

    for index, cell in enumerate(payload["cells"], start=1):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None

    assert "requirements-kaggle.txt" in code
    assert "payload.get('errors')" in code
    assert "synthetic HGT unit smoke" in code
    assert "APPROVED_PROJECT_06_HGT_BENCHMARK" in code
    assert "APPROVED_PROJECT_06_MODEL_BUNDLE" in code
    assert "knetminer_llm.evaluation.cli" in code
    assert "--device" in code
    assert "cuda" in code
    assert "--max-epochs" in code
    assert "100" in code
    assert "--patience" in code
    assert "10" in code


def test_approved_model_download_gates_clear_synthetic_offline_flags() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in payload["cells"]
        if cell["cell_type"] == "code"
    )

    assert code.count("os.environ.pop('HF_HUB_OFFLINE', None)") == 2
    assert code.count("os.environ.pop('TRANSFORMERS_OFFLINE', None)") == 2


def test_kaggle_manifest_pins_a_pascal_compatible_cuda_wheel() -> None:
    requirements = KAGGLE_REQUIREMENTS.read_text(encoding="utf-8")

    assert "--extra-index-url https://download.pytorch.org/whl/cu121" in requirements
    assert "torch==2.5.1+cu121" in requirements
    assert "torchvision==0.20.1+cu121" in requirements
    assert "torchaudio==2.5.1+cu121" in requirements
