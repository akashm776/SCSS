"""Validate report arithmetic against the retained D1 comparison."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.build_research_summary import (
    load_comparison,
    render_tables,
    validate_aggregate,
)


def test_tables_match_published_results():
    expected = Path(__file__).resolve().parents[1] / "docs/paper/detailed-results.md"
    assert render_tables(load_comparison()) == expected.read_text()


def test_rejects_inconsistent_aggregate():
    comparison = deepcopy(load_comparison())
    comparison["timecourse"]["50"]["effects"]["w100"]["H_Select"]["loss"] += 0.01
    with pytest.raises(ValueError, match="Aggregate mismatch: w100/H_Select/loss"):
        validate_aggregate(comparison)
