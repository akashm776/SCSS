"""Population coverage, inherited calibration, and live head/gradient checks."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src import clip_placement_v2d as population
from src.clip_placement_evidence import Journal
from src.clip_placement_preflight import ContractViolation, InvalidEvidence
from tests.test_clip_placement_preflight import features, preserve_test_environment

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def p():
    return json.loads((ROOT / "configs/clip_placement_preflight.json").read_text())


@pytest.fixture
def spec():
    return json.loads((ROOT / "configs/clip_placement_v2d.json").read_text())


def test_calibration_reference_preserves_all_24_coefficients(spec):
    summary, cells = population.calibration_reference(ROOT, spec)
    assert summary["status"] == "PASS"
    assert len(cells) == 6
    assert sum(len(c["coefficients"]) for c in cells.values()) == 24
    assert all(c["optimizer_updates"] == 0 for c in cells.values())


def test_coverage_requires_every_prescribed_batch(p):
    keys = population.expected_keys(p)
    assert len(keys) == 624
    assert sum("/None/" in k for k in keys) == 24
    rows = {k: {"status": "PASS"} for k in keys}
    assert population.completion(rows, p) == "PASS"
    omitted = rows.pop(next(iter(rows)))
    assert population.completion(rows, p) == "INCOMPLETE"
    rows[next(iter(rows))] = {"status": "FAIL"}
    assert population.completion(rows, p) == "FAIL"
    rows["unexpected"] = omitted
    with pytest.raises(InvalidEvidence):
        population.completion(rows, p)


def test_actual_batch_forward_vjps_and_state_are_preserved(p, spec):
    model, opt, sched, batch, cfg, record, _ = features()
    for v in model.parameters():
        v.grad = torch.ones_like(v)
    before = g.stamp(model, opt, sched)
    evidence = population.audit_batch(
        model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
    )
    assert g.stamp(model, opt, sched) == before
    assert evidence["preservation"]["preserved"]
    assert evidence["diagnostic_on_off_identical"]
    assert evidence["placement_contract"]["status"] == "PASS"
    assert len(evidence["feature_gradient_contract"]["per_query"]) == len(
        record["eligible_queries"]
    )
    assert evidence["optimizer_updates"] == 0
    assert "auxiliary_feature_gradients" in evidence["feature_gradient_contract"]


def test_calibration_forward_must_match_exactly(p, spec):
    model, opt, sched, batch, cfg, record, _ = features()
    common = d.rng_state()
    evidence = population.audit_batch(
        model, opt, sched, batch, cfg, record, 1, common, 0.02, p, spec
    )
    archived = dict(evidence, eligible_queries=record["eligible_queries"])
    population.verify_calibration_forward(evidence, archived, record)
    archived["native_loss"] += 1e-7
    with pytest.raises(InvalidEvidence, match="calibrated treatment"):
        population.audit_batch(
            model, opt, sched, batch, cfg, record, 1, common, 0.02, p, spec, archived
        )


def test_empty_batch_keeps_native_and_has_no_scale_gradient(p, spec):
    model, opt, sched, batch, cfg, record, _ = features()
    record = deepcopy(record)
    record["eligible_queries"] = []
    evidence = population.audit_batch(
        model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
    )
    assert evidence["empty"]
    assert evidence["native_loss"] > 0
    assert evidence["auxiliary_gradient_norm"] == 0
    assert evidence["gradient_cosine"] is None
    assert evidence["placements"] == {}
    assert all(v == 0 for v in evidence["auxiliary_losses"].values())


def test_forward_violation_stops_before_gradients_and_preserves_evidence(
    p, spec, monkeypatch
):
    model, opt, sched, batch, cfg, record, _ = features()
    before = g.stamp(model, opt, sched)
    monkeypatch.setattr(
        population,
        "placement_contract",
        lambda *args: {"status": "FAIL", "failures": ["injected"]},
    )

    def forbidden(*args):
        raise AssertionError("Gradient probe called after failed geometry")

    monkeypatch.setattr(population, "gradients", forbidden)
    with pytest.raises(ContractViolation) as caught:
        population.audit_batch(
            model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
        )
    assert caught.value.evidence["placements"]
    assert caught.value.evidence["preservation"]["preserved"]
    assert g.stamp(model, opt, sched) == before


def test_operational_error_restores_state_without_scientific_failure(
    p, spec, monkeypatch
):
    model, opt, sched, batch, cfg, record, _ = features()
    before = g.stamp(model, opt, sched)

    def device_error(*args):
        with torch.no_grad():
            next(model.parameters()).add_(1)
        raise RuntimeError("simulated device interruption")

    monkeypatch.setattr(population, "gradients", device_error)
    with pytest.raises(RuntimeError, match="device interruption"):
        population.audit_batch(
            model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
        )
    assert g.stamp(model, opt, sched) == before


def test_recovery_keeps_prior_attempt_and_rejects_contract_changes(tmp_path, p):
    journal = Journal(tmp_path, {"stage": "v2d", "gap": 0.02})
    key = sorted(population.expected_keys(p))[0]
    journal.put(key, {"status": "PASS"})
    interrupted_attempt = journal.attempt
    resumed = Journal(tmp_path, {"stage": "v2d", "gap": 0.02})
    assert interrupted_attempt + ".json" in resumed.prior_attempts
    assert resumed.get(key) == {"status": "PASS"}
    assert resumed.events[-1]["event"] == "reuse"
    with pytest.raises(InvalidEvidence):
        Journal(tmp_path, {"stage": "v2d", "gap": 0.03})


def test_new_observation_reproduces_the_existing_calibration_path(p, spec):
    from src.clip_placement_v2c import calibration

    model, opt, sched, batch, cfg, record, _ = features()
    common = d.rng_state()
    archived = calibration(
        model, opt, sched, [batch] * 4, [record] * 4, cfg, 1, common, 0.02, p
    )
    evidence = population.audit_batch(
        model,
        opt,
        sched,
        batch,
        cfg,
        record,
        1,
        common,
        0.02,
        p,
        spec,
        archived["unweighted"][0],
    )
    assert evidence["calibration_forward_reproduced"]


@pytest.mark.parametrize(
    "mutation",
    ["margin", "gradient", "scale", "precision", "preservation", "missing_query"],
)
def test_independent_saved_evidence_audit_rejects_corruption(p, spec, mutation):
    from scripts.audit_clip_placement_v2d import validate_evidence

    model, opt, sched, batch, cfg, record, _ = features()
    evidence = population.audit_batch(
        model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
    )
    assert validate_evidence(
        evidence, record["eligible_queries"], p, 0.02, spec["gradient_contract"]
    ) == 9 * len(record["eligible_queries"])
    grads = evidence["feature_gradient_contract"]
    query = next(iter(grads["per_query"]))
    if mutation == "margin":
        evidence["placements"]["Wrong"]["r_train"][0] = 0.02
    elif mutation == "gradient":
        grads["per_query"][query]["Wrong"]["source"]["relative_vector_error"] = 0.1
    elif mutation == "scale":
        grads["scale"]["Boundary"]["partition"] = [False, False]
    elif mutation == "precision":
        evidence["fp32_settings"]["matmul_tf32"] = True
    elif mutation == "preservation":
        evidence["preservation"]["after_sha256"] = "changed"
    else:
        del grads["per_query"][query]
    with pytest.raises(ValueError):
        validate_evidence(
            evidence, record["eligible_queries"], p, 0.02, spec["gradient_contract"]
        )


def test_issued_bundle_matches_sources_and_keeps_old_manifests():
    import ast
    import base64
    import hashlib
    import zlib

    path = ROOT / "colabs/clip_placement_v2d_drive_one_cell.py"
    fields = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("BASE_COMMIT", "BUNDLE_SHA256", "PAYLOAD"):
                fields[node.targets[0].id] = ast.literal_eval(node.value)
    raw = zlib.decompress(base64.b64decode(fields["PAYLOAD"]))
    assert hashlib.sha256(raw).hexdigest() == fields["BUNDLE_SHA256"]
    sources = json.loads(raw)
    manifest = json.loads(sources["placement_v2d_source_manifest.json"])
    for name, sha in manifest.items():
        assert sources[name] == (ROOT / name).read_text()
        assert hashlib.sha256(sources[name].encode()).hexdigest() == sha
    for name in (
        "placement_source_manifest.json",
        "placement_v2a_source_manifest.json",
        "placement_v2a_py31316_source_manifest.json",
        "placement_v2b_source_manifest.json",
        "placement_v2c_source_manifest.json",
    ):
        frozen = json.loads((ROOT / name).read_text())
        for dependency, sha in frozen.items():
            assert hashlib.sha256((ROOT / dependency).read_bytes()).hexdigest() == sha
    assert (
        fields["BASE_COMMIT"]
        == json.loads(
            (ROOT / "configs/clip_placement_v2a_reference/run_config.json").read_text()
        )["repository_commit"]
    )
    compile(path.read_text(), str(path), "exec")


def test_diagnostics_cannot_mutate_shared_candidate_tensors(p, spec, monkeypatch):
    model, opt, sched, batch, cfg, record, _ = features()
    original = population.gradients

    def mutate(*args):
        measured = original(*args)
        with torch.no_grad():
            args[1]["Wrong"]["z"].add_(0.01)
        return measured

    monkeypatch.setattr(population, "gradients", mutate)
    with pytest.raises(AssertionError, match="live scoring inputs"):
        population.audit_batch(
            model, opt, sched, batch, cfg, record, 1, d.rng_state(), 0.02, p, spec
        )
