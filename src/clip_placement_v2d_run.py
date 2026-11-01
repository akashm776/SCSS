"""Restore authenticated D1 states and audit the complete initial exposure grid."""

import gc
import json
from pathlib import Path
import torch
from src import clip_checkpoint_diagnostic as d
from src import clip_geometry_audit as g
from src import clip_train
from src import clip_weight_lr as wl
from src.clip_geometry_run import authenticate_states
from src.clip_source_regime import (
    load_protocol,
    validate_population,
    validate_batch_identity,
)
from src.clip_source_resume_store import safe_load, expand
from src.clip_placement_evidence import checksum, checked_path
from src.clip_placement_preflight import InvalidEvidence, ContractViolation
from src.clip_placement_v2d import audit_batch, expected_keys


def run(d1, inputs, p, delta, spec, calibration_cells, journal):
    """Yield immutable check results; stop at first demonstrated contract violation.

    Remaining checks are explicitly unperformed. Each observation restores the
    starting state; an interrupted observation is recomputed in full on resume.
    """
    from transformers import AutoProcessor

    if not set(journal.index) <= expected_keys(p):
        raise InvalidEvidence("Unexpected committed population check")
    frozen = load_protocol()
    population = json.loads((d.ROOT / p["population"]).read_text())
    if population != inputs["frozen_population.json"]:
        raise InvalidEvidence("D1 treatment population differs")
    plan = json.loads((d.ROOT / p["plan"]).read_text())
    grid = validate_population(population, frozen, plan)
    if inputs["common_branch_plans.json"] != plan["continuation_batches"]:
        raise InvalidEvidence("Continuation plans differ")
    for seed in p["training_seeds"]:
        clip_train.set_global_seed(seed)
        cfg = inputs[f"seed_{seed}/training_config.json"]
        # Only relocate the authenticated exclusion file; all numerical settings stay unchanged.
        config = json.loads(json.dumps(cfg))
        config["dataset"]["diagnostic_holdout_indices"] = str(
            (Path(d1) / "training_exclusion_indices.json").resolve()
        )
        if config["training"]["mixed_precision"] != "bf16":
            raise InvalidEvidence("Expected D1 bf16 numerical path")
        device, _ = clip_train.resolve_device(config["runtime"])
        processor = AutoProcessor.from_pretrained(config["model"]["checkpoint"])
        data = clip_train.build_clip_training_data(config, processor, pin_memory=False)
        if data.train_dataset.source_indices != inputs["split_plan.json"]["training"]:
            raise InvalidEvidence("Dataset training identities changed")
        model = d.CLIPEncoderBackend.from_pretrained(config["model"]["checkpoint"]).to(
            device
        )
        trainable = d.configure_clip_trainable_parameters(
            model, config["model"]["trainable_policy"]
        )
        if (
            trainable["trainable_parameter_count"],
            trainable["trainable_tensor_count"],
        ) != (10895617, 35):
            raise InvalidEvidence("Trainable policy differs")
        opt = d.build_clip_optimizer(model, config["optimizer"])
        sched = clip_train.build_scheduler(
            opt,
            warmup_steps=config["scheduler"]["warmup_steps"],
            total_steps=inputs["runtime_protocol.json"]["scheduler_horizon"],
        )
        pointer = inputs[f"resume_state/seed_{seed}_trunk.json"]
        entry = pointer["generations"][0]
        blob = checked_path(d1, "resume_state/" + entry["file"])
        if checksum(blob) != entry["sha256"]:
            raise InvalidEvidence("Trunk changed after input authentication")
        saved = safe_load(blob)
        if (
            saved["contract"] != pointer["contract"]
            or saved["key"] != f"seed_{seed}_trunk"
        ):
            raise InvalidEvidence("Decoded trunk identity differs")
        states = {
            w: expand(saved["payload"]["snapshots"][str(w)], model)
            for w in p["checkpoint_steps"]
        }
        del saved
        try:
            authenticate_states(
                states,
                plan["seeds"][str(seed)],
                p["checkpoint_steps"],
                inputs[f"seed_{seed}/trunk_audit.json"],
            )
        except AssertionError as error:
            raise InvalidEvidence(
                "Native starting state authentication failed"
            ) from error
        common = states[100]
        groups = [
            (stream, plan["continuation_batches"][f"{seed}:{stream}"])
            for stream in p["stream_seeds"]
        ]
        groups.append((None, plan["calibration_batches"][str(seed)]))
        for stream, indices in groups:
            flat = [i for batch in indices for i in batch]
            with g.preserved(model, opt, sched):
                batches, identities = d.build_batches(
                    data, flat, seed=seed, epoch=common["epoch"], canonical=False
                )
            if stream is None:
                expected = inputs[f"seed_{seed}/calibration_batches.json"]
                identity = dict(
                    records=identities,
                    caption_epoch=common["epoch"],
                    source_indices=flat,
                    disjoint_from_report_probes_and_continuation_by_image_key=True,
                )
            else:
                expected = inputs[f"seed_{seed}/stream_{stream}/training_batches.json"]
                identity = dict(records=identities, caption_epoch=common["epoch"])
            if identity != expected:
                raise InvalidEvidence("Processed batch/caption identity differs")
            records = [grid[seed, stream, t] for t in range(len(batches))]
            for offset, record in enumerate(records):
                validate_batch_identity(
                    record, identities[offset * 64 : (offset + 1) * 64]
                )
            for w in p["checkpoint_steps"]:
                wl.prepare_branch(model, opt, sched, states[w], common, common)
                for offset, (batch, record) in enumerate(zip(batches, records)):
                    key = f"geometry/{seed}/{stream}/{w}/{offset}"
                    context = dict(
                        training_seed=seed,
                        stream_seed=stream,
                        weight_step=w,
                        batch_offset=offset,
                        population=record["population"],
                        eligible_queries=record["eligible_queries"],
                    )
                    cached = journal.get(key)
                    if cached is None:
                        try:
                            result = dict(
                                status="PASS",
                                context=context,
                                evidence=audit_batch(
                                    model,
                                    opt,
                                    sched,
                                    batch,
                                    config,
                                    record,
                                    w,
                                    common,
                                    delta,
                                    p,
                                    spec,
                                    (
                                        calibration_cells[seed, w]["unweighted"][offset]
                                        if stream is None
                                        else None
                                    ),
                                ),
                            )
                        except ContractViolation as error:
                            result = dict(
                                status="FAIL",
                                context=context,
                                violation=error.record,
                                evidence=getattr(error, "evidence", {}),
                            )
                        journal.put(key, result)
                    else:
                        result = cached
                    if result["context"] != context:
                        raise InvalidEvidence(
                            "Cached check has a different exposure context"
                        )
                    yield key, result
                    if result["status"] == "FAIL":
                        return
            del batches, identities
            gc.collect()
        del states, common, model, opt, sched, data
        gc.collect()
        torch.cuda.empty_cache()
