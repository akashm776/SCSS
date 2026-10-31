"""D2-P only. No reporting outcomes, preparation training, or D2 launch."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TOKENIZERS_PARALLELISM']='false'


def numerical_environment(reference):
    import torch
    from src.clip_placement_preflight import InvalidEvidence
    required=reference['run_manifest.json']['contract']
    versions={name:importlib.metadata.version(name) for name in required['packages']}
    if versions!=required['packages']:raise InvalidEvidence('Pinned D1 package versions differ: '+str(versions))
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('D2-P requires one A100 CUDA GPU; no CPU fallback')
    if torch.version.cuda!=required['cuda_version']:raise InvalidEvidence('CUDA build differs from D1')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader,nounits'],text=True).strip()
    if gpu!=required['gpu']:raise InvalidEvidence('GPU contract differs: '+gpu)
    if list(sys.version_info[:3])!=required['python_version']:raise InvalidEvidence('Python version differs from D1')
    if os.environ['CUBLAS_WORKSPACE_CONFIG']!=required['cublas_workspace']:raise InvalidEvidence('CUBLAS configuration differs')
    ram=next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemTotal:'))
    if ram<20*1024**3:raise RuntimeError('High-RAM runtime with >=20 GiB host memory required')
    return dict(packages=versions,python=sys.version,gpu=gpu,cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        precision='D1 bf16, no GradScaler',host_ram_bytes=ram)


def main(d1,output,inputs_only=False):
    from src.clip_placement_evidence import (checksum,read_checked,validate_gap,authenticate_inputs,Journal,digest)
    from src.clip_placement_preflight import InvalidEvidence, acceptance
    from src.clip_source_resume_store import atomic_json
    from colabs.clip_source_lease_repair import install
    output=Path(output).resolve();d1=Path(d1).resolve()
    if output==d1 or output.is_relative_to(d1) or d1.is_relative_to(output):
        raise ValueError('D2-P output must be separate from the read-only D1 input')
    expected=630;completed=0;violations=[];valid=True;complete=False;error=None;journal=None
    # Reuse the repaired lease implementation only on the NEW D2-P output tree.
    with install()(output) as check:
        attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        try:
            source_manifest=json.loads((ROOT/'placement_source_manifest.json').read_text())
            for name,sha in source_manifest.items():
                if checksum(ROOT/name)!=sha:raise InvalidEvidence('Implementation/configuration changed: '+name)
            p=json.loads((ROOT/'configs/clip_placement_preflight.json').read_text())
            gap=json.loads((ROOT/p['gap_manifest']).read_text());delta=validate_gap(p,gap)
            ref=json.loads((ROOT/'configs/clip_placement_inputs.json').read_text())
            print('D2P_AUTHENTICATING_READ_ONLY_D1_INPUTS:',d1,flush=True)
            inputs=authenticate_inputs(d1,ref)
            if inputs_only:
                result=dict(status='INCOMPLETE',reason='Input authentication only; GPU checks not executed',
                    delta=delta,gap_sha256=checksum(ROOT/p['gap_manifest']),input_reference_sha256=checksum(ROOT/'configs/clip_placement_inputs.json'),
                    launch_eligible=False,d2_launched=False,optimizer_updates=0)
                atomic_json(output/'input_authentication.json',result)
                print(json.dumps(result),flush=True)
                return result
            environment=numerical_environment(inputs)
            commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
            contract=dict(experiment=p['experiment'],design=p,design_sha256=checksum(ROOT/'configs/clip_placement_preflight.json'),
                gap_manifest_sha256=checksum(ROOT/p['gap_manifest']),delta=delta,input_reference=ref,
                implementation=source_manifest,repository_commit=commit,
                environment={k:v for k,v in environment.items() if k!='host_ram_bytes'},
                acceptance_policy='stop on first demonstrated contract violation; remaining checks explicitly unperformed')
            journal=Journal(output,contract,check)
            atomic_json(output/'environment.json',environment)
            from src.clip_placement_preflight_run import run
            for key,result in run(d1,inputs,p,delta,journal):
                completed+=1
                if result['status']=='FAIL':violations.append(dict(check=key,context=result['context'],**result['violation']))
                print('D2P_CHECK:',key,result['status'],f'{completed}/{expected}',flush=True)
                record=acceptance(valid=True,complete=False,violations=violations,expected=expected,completed=completed)
                atomic_json(output/'acceptance.json',dict(**record,attempt=journal.attempt,contract_sha256=journal.contract_hash))
            complete=completed==expected
        except InvalidEvidence as exc:
            valid=False;error=dict(kind='provenance_or_configuration',type=type(exc).__name__,message=str(exc))
            traceback.print_exc()
        except BaseException as exc:
            error=dict(kind='operational_or_undiagnosed_numerical',type=type(exc).__name__,message=str(exc))
            traceback.print_exc()
        result=acceptance(valid=valid,complete=complete,violations=violations,expected=expected,completed=completed,error=error)
        result.update(attempt=journal.attempt if journal else attempt,
            contract_sha256=journal.contract_hash if journal else None,
            limitation='Initial execution-contract audit only; not future feasibility or scientific utility.')
        if journal:
            result['committed_check_index_sha256']=checksum(output/'index.json') if (output/'index.json').exists() else None
            journal.finish(result['status'],error)
        else:
            atomic_json(output/'startup_attempts'/(attempt+'.json'),result)
        atomic_json(output/'acceptance.json',result)
        print('D2P_ACCEPTANCE:',json.dumps(result),flush=True)
        print('D2_TRAINING_NOT_LAUNCHED',flush=True)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--d1-run',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--inputs-only',action='store_true')
    args=parser.parse_args();result=main(args.d1_run,args.output,args.inputs_only)
    input_success=args.inputs_only and result.get('reason')=='Input authentication only; GPU checks not executed'
    if result['status']!='PASS' and not input_success:raise SystemExit(2)
