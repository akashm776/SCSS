"""v2b first-batch gradient validation, Python 3.13.16. Never launch training."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TOKENIZERS_PARALLELISM']='false'


def main(d1, output, inputs_only=False):
    from src.clip_placement_evidence import checksum, validate_gap, authenticate_inputs, Journal
    from src.clip_placement_preflight import InvalidEvidence
    from src.clip_placement_v2a import reference
    from src.clip_source_resume_store import atomic_json
    from colabs.clip_source_lease_repair import install
    from src.clip_placement_v2a_environment import capture_environment, validate_environment
    output, d1 = Path(output).resolve(), Path(d1).resolve()
    if output == d1 or output.is_relative_to(d1) or d1.is_relative_to(output):
        raise ValueError('Output must be separate from read-only D1 inputs')
    if (output/'run_config.json').exists():
        previous=json.loads((output/'run_config.json').read_text())
        if previous.get('experiment')!='clip_placement_v2b_gradients':
            raise ValueError('Refusing to write into a different experiment, including v1')
    # Every invocation is a separate attempt, never overwrite an old receipt.
    with install()(output) as check:
        status='INCOMPLETE';error=None;journal=None;record=None
        attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        try:
            manifest=json.loads((ROOT/'placement_v2b_source_manifest.json').read_text())
            for name, sha in manifest.items():
                if checksum(ROOT/name) != sha: raise InvalidEvidence('Implementation changed: '+name)
            spec=json.loads((ROOT/'configs/clip_placement_v2b.json').read_text())
            p=json.loads((ROOT/spec['original_design']).read_text())
            original, archived=reference(ROOT,spec)
            from src.clip_placement_v2b import forward_reference
            parent,parent_audit=forward_reference(ROOT,spec)
            atomic_json(output/'forward_parent_audit.json',parent_audit)
            delta=validate_gap(p,json.loads((ROOT/p['gap_manifest']).read_text()))
            if delta != original['delta'] or p != original['design']:
                raise InvalidEvidence('Frozen v1 gap/design changed')
            ref=json.loads((ROOT/'configs/clip_placement_inputs.json').read_text())
            if ref != original['input_reference']: raise InvalidEvidence('D1 input lineage changed')
            print('V2B_AUTHENTICATING_READ_ONLY_D1_INPUTS:',d1,flush=True)
            inputs=authenticate_inputs(d1,ref)
            if inputs_only:
                result=dict(status='INCOMPLETE',input_authentication='PASS',
                    reason='No GPU diagnostic performed',launch_eligible=False,d2_launched=False,
                    optimizer_updates=0,delta=delta,reference_files=spec['reference_files'])
                atomic_json(output/'input_authentication_attempts'/(attempt+'.json'),result)
                return result
            environment=capture_environment(original['environment'])
            atomic_json(output/'environment_attempts'/(attempt+'.json'),environment)
            print('V2B_OBSERVED_ENVIRONMENT:',json.dumps(environment),flush=True)
            validate_environment(environment,original['environment'],spec['environment_revision'])
            contract=dict(experiment=spec['experiment'],specification=spec,original_design=p,
                delta=delta,implementation=manifest,environment={k:v for k,v in environment.items() if k!='host_ram_bytes'},
                original_environment=original['environment'],
                original_contract_sha256=json.loads((ROOT/spec['reference_root']/'acceptance.json').read_text())['contract_sha256'],
                input_reference=ref,repository_commit=original['repository_commit'])
            # Authenticate the actual checkout as well as the overlay.
            import subprocess
            if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=original['repository_commit']:
                raise InvalidEvidence('Repository base differs from archived execution')
            journal=Journal(output,contract,check)
            key='gradients/271828/2026100101/100/0'
            record=journal.get(key)
            if record is None:
                from src.clip_placement_v2b_run import run
                record=run(d1,inputs,p,delta,spec,archived,parent)
                journal.put(key,record)
            status=record['status']
        except InvalidEvidence as exc:
            status='INVALID';error=dict(kind='provenance_or_configuration',message=str(exc))
            traceback.print_exc()
        except BaseException as exc:
            status='INCOMPLETE';error=dict(kind='operational_or_undiagnosed_numerical',type=type(exc).__name__,message=str(exc))
            traceback.print_exc()
        receipt=dict(status=status,attempt=journal.attempt if journal else attempt,
            scope='First-batch gradient validation only; not calibration or full D2-P acceptance',
            launch_eligible=False,d2_launched=False,optimizer_updates=0,gradient_probes=record.get('gradient_probes',0) if record else 0,
            calibration_batches=0,next_stage='Pressure calibration' if status=='PASS' else None,
            operational_error=error,reason=record.get('reason') if record else None,
            contract_sha256=journal.contract_hash if journal else None,
            committed_check_index_sha256=checksum(output/'index.json') if journal and (output/'index.json').exists() else None)
        if journal: journal.finish(status,error)
        atomic_json(output/'receipts'/((journal.attempt if journal else attempt)+'.json'),receipt)
        atomic_json(output/'acceptance.json',receipt)
        print('V2B_ACCEPTANCE:',json.dumps(receipt),flush=True)
        print('NO_OPTIMIZER_UPDATES_NO_CALIBRATION_NO_D2_TRAINING',flush=True)
        return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--d1-run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--inputs-only',action='store_true')
    args=parser.parse_args()
    result=main(args.d1_run,args.output,args.inputs_only)
    if result['status']!='PASS' and result.get('input_authentication')!='PASS': raise SystemExit(2)
