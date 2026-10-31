"""Regression and rejection checks against the archived calibration evidence."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.audit_clip_placement_v2a import canonical, read_input
from scripts.audit_clip_placement_v2c import audit, report

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiment_results/clip_placement_v2c_2026-10-31"


@pytest.fixture(scope="module")
def archive_records():
    read = read_input(ARCHIVE / "source.zip")
    names = ["acceptance.json", "run_config.json", "index.json"]
    index = json.loads(read("index.json"))
    names.extend(entry["file"] for entry in index.values())
    return {name: json.loads(read(name)) for name in names}


def write_records(directory, records):
    """Rehash test mutations so rejection exercises validation, not checksums."""
    for entry in records["index.json"].values():
        entry["sha256"] = canonical(records[entry["file"]])
    records["acceptance.json"]["committed_check_index_sha256"] = canonical(
        records["index.json"]
    )
    for name, value in records.items():
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )


def test_archived_output_is_unchanged():
    result = audit(ARCHIVE / "source.zip")
    expected = json.loads((ARCHIVE / "audit.json").read_text())
    expected.pop("source_zip_sha256")
    assert result == expected
    assert report(result) == (ARCHIVE / "REPORT.md").read_text()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("pressure", "Coefficient or verification summary"),
        ("margin", "Actual placement failed"),
        ("coverage", "Incomplete exposure arrays"),
        ("state", "State preservation failed"),
        ("scale", "Missing arms or scale leakage"),
        ("precision", "FP32 context differs"),
        ("replay", "Weighted replay differs"),
    ],
)
def test_rejects_rehashed_contract_violations(
    tmp_path, archive_records, mutation, message
):
    records = deepcopy(archive_records)
    entry = next(iter(records["index.json"].values()))
    evidence = records[entry["file"]]["result"]["evidence"]
    batch = evidence["unweighted"][0]

    if mutation == "pressure":
        evidence["coefficients"]["Wrong"]["coefficient"] *= 2
    elif mutation == "margin":
        batch["placements"]["Wrong"]["g_logit"][0] += 1
    elif mutation == "coverage":
        batch["placements"]["Wrong"]["r_train"].pop()
    elif mutation == "state":
        evidence["preservation"][0]["preserved"] = False
    elif mutation == "scale":
        batch["scale_disconnected"]["Wrong"] = False
    elif mutation == "precision":
        batch["fp32_settings"]["matmul_tf32"] = True
    elif mutation == "replay":
        evidence["verification"][0]["native_loss"] += 1

    write_records(tmp_path, records)
    with pytest.raises(ValueError, match=message):
        audit(tmp_path)


def test_rejects_missing_calibration_cell(tmp_path, archive_records):
    records = deepcopy(archive_records)
    records["index.json"].pop(next(iter(records["index.json"])))
    write_records(tmp_path, records)
    with pytest.raises(ValueError, match="Incomplete cell grid"):
        audit(tmp_path)
