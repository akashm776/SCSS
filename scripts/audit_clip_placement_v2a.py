"""Authenticate and summarize the reported v2a PASS; no model or gradient execution."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = '9b26e816ed8eba4ef2e661e6536ef3604eee665c7a57568470e171d47629c918'
INDEX = '164a2219dc12987d9daaa77042970a31405a6e925cc97ece93e177d6dea785fd'
KEY = 'paired_forward/271828/2026100101/100/0'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_input(path):
    """Read only four referenced files. ZIP contents are never extracted."""
    path = Path(path).resolve()
    if path.is_dir():
        def read(name):
            target = (path/name).resolve()
            require(target.is_relative_to(path), 'Unsafe artifact path')
            return target.read_bytes()
        return read
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        candidates = [n for n in names if n.endswith('/acceptance.json') or n == 'acceptance.json']
        require(len(candidates)==1, 'Expected exactly one acceptance.json in ZIP')
        prefix = candidates[0][:-len('acceptance.json')]
        require(len(names)==len(set(names)), 'Duplicate ZIP member names')
    def read(name):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe artifact path')
        with zipfile.ZipFile(path) as archive:
            return archive.read(prefix+name)
    return read


def stats(values):
    values = list(values)
    require(bool(values) and all(math.isfinite(v) for v in values), 'Nonfinite or empty measurements')
    return dict(count=len(values),minimum=min(values),maximum=max(values),
                mean=statistics.fmean(values),median=statistics.median(values),
                maximum_absolute=max(map(abs,values)))


def flatten(value):
    if isinstance(value,list):
        return [x for item in value for x in flatten(item)]
    return [value]


def numerical_audit(evidence, design, delta, ids):
    """Independently recompute contracts from saved observations, not status flags."""
    placements = evidence['v2a']['placements']
    require(set(placements)=={'Wrong','Boundary','Correct'}, 'Missing placement arm')
    arms = {}
    for arm, sign in [('Wrong',-1),('Boundary',0),('Correct',1)]:
        v = placements[arm]
        for field in ['r_geom','r_train','g_logit','m_loss','p_sim','ell_plus','s_aux','e_aux','A']:
            require(len(v[field])==len(ids), 'Wrong exposure coverage: '+arm+'/'+field)
            stats(v[field])
        tau = v['tau']
        require(math.isfinite(tau) and tau>0 and v['delta']==delta and v['target_margin']==sign*delta,
                'Wrong target or scale')
        geometry = [abs(r-sign*delta) for r in v['r_geom']]
        similarity = [abs(r-sign*delta) for r in v['r_train']]
        logit = [abs(r-tau*sign*delta) for r in v['g_logit']]
        limits = dict(geometry=max(design['fp32_margin_atol'],design['fp32_margin_rtol_delta']*delta),
            similarity=design['training_margin_rtol_delta']*delta,
            logit=design['training_logit_rtol_scaled_delta']*tau*delta)
        failures = {name:[q for q,x in zip(ids,errors) if x>limits[name]]
                    for name,errors in [('geometry',geometry),('similarity',similarity),('logit',logit)]}
        failures['similarity_sign'] = [q for q,x in zip(ids,v['r_train']) if sign and x*sign<=0]
        failures['logit_sign'] = [q for q,x in zip(ids,v['g_logit']) if sign and x*sign<=0]
        require(not any(failures.values()), 'Saved observations violate v2a contracts: '+str(failures))
        arms[arm] = dict(tau=tau,maximum_geometry_error=max(geometry),
            maximum_similarity_error=max(similarity),maximum_logit_error=max(logit),
            limits=limits,failures=failures,
            similarity_margin=stats(v['r_train']),logit_gap=stats(v['g_logit']))
    spacing = {}
    for left,right in [('Wrong','Boundary'),('Boundary','Correct')]:
        gaps = [b-a for a,b in zip(placements[left]['r_train'],placements[right]['r_train'])]
        require(min(gaps)>=design['minimum_adjacent_separation_delta']*delta, 'Spacing contract violated')
        spacing[left+'/'+right] = stats(gaps)
    before,after = evidence['v1']['native'],evidence['v2a']['native']
    require(set(before)==set(after), 'Native field sets differ')
    for head in [before,after]:
        for name in ['similarity','logits']:
            require(len(head[name])==64 and all(len(row)==64 for row in head[name]), 'Native matrix is not 64x64')
        for name in ['p_sim','ell_plus','A_row','A_col','loss_row','loss_col']:
            require(len(head[name])==64, 'Missing full native batch measurements')
    differences = {}
    for field in before:
        old,new = flatten(before[field]),flatten(after[field])
        require(len(old)==len(new), 'Native measurement shape changed: '+field)
        differences[field] = stats(b-a for a,b in zip(old,new))
    return dict(arms=arms,spacing=spacing,native_differences=differences,
        native_loss_before=before['loss_native'],native_loss_after=after['loss_native'],
        native_loss_change=after['loss_native']-before['loss_native'])


def audit(path):
    read = read_input(path)
    contents = {name:read(name) for name in ['acceptance.json','run_config.json','index.json']}
    acceptance,contract,index = (json.loads(contents[name]) for name in ['acceptance.json','run_config.json','index.json'])
    require(acceptance['status']=='PASS', 'No passing receipt')
    require(acceptance['contract_sha256']==CONTRACT==canonical(contract), 'Wrong run contract')
    require(acceptance['committed_check_index_sha256']==INDEX==sha(contents['index.json']), 'Wrong committed index')
    require(set(index)=={KEY}, 'Unexpected observation coverage')
    entry = index[KEY];contents[entry['file']] = read(entry['file'])
    require(sha(contents[entry['file']])==entry['sha256'], 'Corrupt paired record')
    record = json.loads(contents[entry['file']])
    require(record['contract_sha256']==CONTRACT and record['key']==KEY, 'Paired record belongs to another run')
    result = record['result'];evidence = result['evidence']
    spec = json.loads((ROOT/'configs/clip_placement_v2a_py31316.json').read_text())
    require(contract['specification']==spec, 'Wrong diagnostic specification')
    require(contract['implementation']==json.loads((ROOT/'placement_v2a_py31316_source_manifest.json').read_text()),
            'Wrong implementation manifest')
    for name,h in contract['implementation'].items():
        require(sha((ROOT/name).read_bytes())==h, 'Local issued dependency changed: '+name)
    require(result['status']=='PASS' and result['encoder_forwards']==1, 'Not a completed paired observation')
    for container in [acceptance,result]:
        for field in ['optimizer_updates','gradient_probes','calibration_batches']:
            require(container[field]==0, 'Unexpected work before gradient validation')
    require(not acceptance['launch_eligible'] and not acceptance['d2_launched'], 'Training was incorrectly authorized')
    preservation = result['preservation']
    require(preservation['preserved'] and preservation['before_sha256']==preservation['after_sha256'],
            'State restoration not authenticated')
    old_index = json.loads((ROOT/spec['reference_root']/'index.json').read_text())
    old_entry = next(iter(old_index.values()))
    old_record = json.loads((ROOT/spec['reference_root']/old_entry['file']).read_text())['result']
    require(result['context']==old_record['context'], 'First-batch identity changed')
    require(evidence['v1']['placements']==old_record['evidence']['placements'], 'Archived observables were not exactly reproduced')
    require(evidence['reproduction']['status']=='PASS' and not evidence['reproduction']['differences'], 'Reproduction gate did not pass')
    require(evidence['shared_inputs_unchanged'], 'Shared inputs mutated')
    settings=evidence['fp32_settings']
    require(not settings['autocast_enabled'] and not settings['matmul_tf32'] and settings['matmul_precision']=='highest',
            'Wrong scoring-head arithmetic settings')
    numerical = numerical_audit(evidence,contract['original_design'],contract['delta'],result['context']['eligible_queries'])
    return dict(status='PASS',scope='Offline audit of one completed v2a paired record; no new model execution',
        contract_sha256=CONTRACT,input_sha256={n:sha(b) for n,b in contents.items()},
        context=result['context'],delta=contract['delta'],**numerical,
        limitation='Does not validate gradients, calibration, full-population feasibility or utility.')


def report(value):
    lines=['# Authenticated v2a paired forward audit','',value['scope'],'',
        '| Arm | Max geometry error | Max similarity error | Max logit error |',
        '|---|---:|---:|---:|']
    for arm,row in value['arms'].items():
        lines.append(f"| {arm} | {row['maximum_geometry_error']:.9g} | {row['maximum_similarity_error']:.9g} | {row['maximum_logit_error']:.9g} |")
    lines += ['',f"Native loss: {value['native_loss_before']:.12g} → {value['native_loss_after']:.12g}; change {value['native_loss_change']:+.12g}.",
        '', 'All frozen placement checks passed when independently recomputed from saved observations.',
        'The complete JSON records distributions of native matrix, partition and loss changes.',
        '',value['limitation'],'']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('input');parser.add_argument('--output',required=True)
    args=parser.parse_args();value=audit(args.input)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'audit.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    (output/'REPORT.md').write_text(report(value))
    print(report(value))
