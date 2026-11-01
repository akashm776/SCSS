"""Audit every planned initial D2 exposure. No continuation training is executed."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(d1, output, inputs_only=False):
    from colabs.clip_source_lease_repair import install
    from src.clip_placement_evidence import (
        Journal,
        authenticate_inputs,
        checksum,
        validate_gap,
    )
    from src.clip_placement_preflight import InvalidEvidence
    from src.clip_placement_v2a import reference
    from src.clip_placement_v2a_environment import (
        capture_environment,
        validate_environment,
    )
    from src.clip_placement_v2d import calibration_reference, completion, expected_keys
    from src.clip_source_resume_store import atomic_json

    output, d1 = Path(output).resolve(), Path(d1).resolve()
    if output == d1 or output.is_relative_to(d1) or d1.is_relative_to(output):
        raise ValueError("Output must be separate from read-only D1 inputs")
    if (output / "run_config.json").exists():
        prior = json.loads((output / "run_config.json").read_text())
        if prior.get("experiment") != "clip_placement_v2d_population":
            raise ValueError("Refusing to overwrite another experiment")

    with install()(output) as check:
        status, error, journal, last = "INCOMPLETE", None, None, None
        records = {}
        parent_summary = None
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        try:
            manifest = json.loads(
                (ROOT / "placement_v2d_source_manifest.json").read_text()
            )
            for name, sha in manifest.items():
                if checksum(ROOT / name) != sha:
                    raise InvalidEvidence("Implementation changed: " + name)
            spec = json.loads((ROOT / "configs/clip_placement_v2d.json").read_text())
            p = json.loads((ROOT / spec["original_design"]).read_text())
            if len(expected_keys(p)) != spec["expected_checks"]:
                raise InvalidEvidence("Coverage specification differs")
            original, _ = reference(ROOT, spec)
            parent_summary, cells = calibration_reference(ROOT, spec)
            atomic_json(output / "calibration_parent_audit.json", parent_summary)
            delta = validate_gap(p, json.loads((ROOT / p["gap_manifest"]).read_text()))
            if delta != original["delta"] or p != original["design"]:
                raise InvalidEvidence("Frozen gap/design changed")
            ref = json.loads((ROOT / "configs/clip_placement_inputs.json").read_text())
            if ref != original["input_reference"]:
                raise InvalidEvidence("D1 input lineage changed")
            print("D2P_AUTHENTICATING_READ_ONLY_D1_INPUTS:", d1, flush=True)
            inputs = authenticate_inputs(d1, ref)
            if inputs_only:
                result = dict(
                    status="INCOMPLETE",
                    input_authentication="PASS",
                    reason="No GPU audit performed",
                    launch_eligible=False,
                    d2_launched=False,
                    optimizer_updates=0,
                )
                atomic_json(
                    output / "input_authentication_attempts" / (attempt + ".json"),
                    result,
                )
                return result
            environment = capture_environment(original["environment"])
            atomic_json(
                output / "environment_attempts" / (attempt + ".json"), environment
            )
            print("D2P_OBSERVED_ENVIRONMENT:", json.dumps(environment), flush=True)
            validate_environment(
                environment, original["environment"], spec["environment_revision"]
            )
            base = subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip()
            if base != original["repository_commit"]:
                raise InvalidEvidence("Repository base differs from archived execution")
            contract = dict(
                experiment=spec["experiment"],
                specification=spec,
                original_design=p,
                delta=delta,
                implementation=manifest,
                environment={
                    k: v for k, v in environment.items() if k != "host_ram_bytes"
                },
                input_reference=ref,
                repository_commit=base,
                calibration_contract_sha256=parent_summary["contract_sha256"],
            )
            journal = Journal(output, contract, check)
            # Freeze coefficients as authenticated outputs, never fit or edit them here.
            coefficients = {
                f"{s}/{w}": cell["coefficients"] for (s, w), cell in cells.items()
            }
            atomic_json(output / "frozen_coefficients.json", coefficients)
            from src.clip_placement_v2d_run import run

            for key, last in run(d1, inputs, p, delta, spec, cells, journal):
                records[key] = last
                print(
                    "D2P_CHECK:",
                    key,
                    last["status"],
                    f"{len(records)}/{len(expected_keys(p))}",
                    flush=True,
                )
                if last["status"] != "PASS":
                    break
            status = completion(records, p)
        except InvalidEvidence as exc:
            status = "INVALID"
            error = dict(kind="provenance_or_configuration", message=str(exc))
            traceback.print_exc()
        except BaseException as exc:
            status = "INCOMPLETE"
            error = dict(
                kind="operational_or_undiagnosed_numerical",
                type=type(exc).__name__,
                message=str(exc),
            )
            traceback.print_exc()

        receipt = dict(
            status=status,
            attempt=journal.attempt if journal else attempt,
            scope="Full initial population under the v2 execution contract; no continuation utility",
            expected_checks=624,
            completed_checks=len(records),
            passing_checks=sum(r["status"] == "PASS" for r in records.values()),
            execution_complete=len(records) == 624,
            launch_eligible=status == "PASS",
            d2_launched=False,
            optimizer_updates=0,
            calibration_refits=0,
            authenticated_coefficients=24 if parent_summary else 0,
            next_stage="D2 continuation training" if status == "PASS" else None,
            operational_error=error,
            violation=last.get("violation") if last else None,
            contract_sha256=journal.contract_hash if journal else None,
            committed_check_index_sha256=(
                checksum(output / "index.json")
                if journal and (output / "index.json").exists()
                else None
            ),
            limitation="Initial coverage only. Runtime assigned-target feasibility remains an online check.",
        )
        if journal:
            journal.finish(status, error)
        atomic_json(output / "receipts" / (receipt["attempt"] + ".json"), receipt)
        atomic_json(output / "acceptance.json", receipt)
        print("D2P_ACCEPTANCE:", json.dumps(receipt), flush=True)
        print("NO_OPTIMIZER_UPDATES_NO_D2_TRAINING", flush=True)
        return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--d1-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    result = main(args.d1_run, args.output, args.inputs_only)
    if result["status"] != "PASS" and result.get("input_authentication") != "PASS":
        raise SystemExit(2)
