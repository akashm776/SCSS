"""Immutable inputs and checksummed, resumable D2-P records (no weight writes)."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import uuid

from src.clip_placement_preflight import InvalidEvidence
from src.clip_source_resume_store import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def checked_path(root,name):
    root=Path(root).resolve();path=(root/name).resolve()
    if not path.is_relative_to(root):raise InvalidEvidence('Path escapes evidence root: '+name)
    return path


def read_checked(root,name,sha):
    path=checked_path(root,name)
    if checksum(path)!=sha:raise InvalidEvidence('Artifact checksum differs: '+name)
    return json.loads(path.read_text())


def authenticate_inputs(root, reference):
    result={}
    for name,sha in reference['files'].items():result[name]=read_checked(root,name,sha)
    for seed in reference['training_seeds']:
        pointer=result[f'resume_state/seed_{seed}_trunk.json']
        entry=pointer['generations'][0]
        if entry!=reference['trunks'][str(seed)]:raise InvalidEvidence('Wrong latest trunk pointer')
        if checksum(checked_path(root,'resume_state/'+entry['file']))!=entry['sha256']:
            raise InvalidEvidence('Wrong/corrupt trunk blob')
    return result


def validate_gap(p,gap):
    for name,actual in [('design_sha256',checksum(ROOT/'configs/clip_placement_preflight.json')),
                        ('population_sha256',checksum(ROOT/p['population'])),
                        ('plan_sha256',checksum(ROOT/p['plan'])),
                        ('generator_sha256',checksum(ROOT/'scripts/build_clip_placement_gap.py'))]:
        if gap[name]!=actual:raise InvalidEvidence('Gap dependency differs: '+name)
    pop=json.loads((ROOT/p['population']).read_text())
    wanted={}
    for b in pop['batches']:
        if b['population']!='continuation':continue
        for w in p['checkpoint_steps']:
            for k,i in enumerate(b['eligible_queries']):
                key=(b['training_seed'],w,b['stream_seed'],b['batch_offset'],i)
                wanted[key]=(b,k)
    by_seed={s:[] for s in p['training_seeds']};seen=set()
    for r in gap['records']:
        key=tuple(r[k] for k in ['training_seed','weight_step','stream_seed','batch_offset','query_index'])
        if key not in wanted or key in seen:raise InvalidEvidence('Wrong/duplicate gap exposure')
        seen.add(key);b,k=wanted[key];state=b['states'][str(r['weight_step'])]
        if any(r[n]!=b['identities'][r['query_index']][n] for n in ['source_index','caption_index','species_id']):
            raise InvalidEvidence('Gap query identity differs')
        if (r['C_index']!=state['supports']['C'][k][0] or r['I_index']!=state['supports']['I'][k][0]
                or r['shard']!=state['shard'] or r['shard_sha256']!=state['shard_sha256']):
            raise InvalidEvidence('Gap source provenance differs')
        if not r['r_C']>0 or not r['r_I']<=0:raise InvalidEvidence('Invalid real C/I bracket')
        half=(r['r_C']-r['r_I'])/2
        if half!=r['half_bracket']:raise InvalidEvidence('Incorrect half bracket')
        by_seed[r['training_seed']].append(half)
    medians={str(s):statistics.median(v) for s,v in by_seed.items()}
    if (seen!=set(wanted) or gap['seed_medians']!=medians or gap['delta']!=statistics.median(medians.values())
            or gap['counts']!={str(s):len(v) for s,v in by_seed.items()} or gap['calibration_included'] is not False):
        raise InvalidEvidence('Gap derivation/coverage differs')
    return gap['delta']


class Journal:
    """One independently restored starting-state check per immutable record.

    An interrupted record is recomputed from authenticated inputs. Completed
    records are reused only under the identical run contract and verified hashes.
    Every invocation has a separate attempt record, including killed invocations.
    Caller holds a single-writer lease for the entire lifetime.
    """
    def __init__(self,root,contract,check=lambda:None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.check=check;self.contract=contract;self.contract_hash=digest(contract)
        path=self.root/'run_config.json'
        if path.exists():
            if json.loads(path.read_text())!=contract:raise InvalidEvidence('D2-P run configuration changed')
        else:atomic_json(path,contract)
        index=self.root/'index.json'
        self.index=json.loads(index.read_text()) if index.exists() else {}
        self.attempt=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')+'_'+uuid.uuid4().hex[:8]
        self.events=[]
        self.prior_attempts={f.name:checksum(f) for f in sorted((self.root/'attempts').glob('*.json'))} if (self.root/'attempts').exists() else {}
        self.finish('RUNNING')

    def finish(self,status,error=None):
        self.check()
        atomic_json(self.root/'attempts'/(self.attempt+'.json'),dict(attempt=self.attempt,status=status,
            contract_sha256=self.contract_hash,events=self.events,prior_attempts=self.prior_attempts,error=error))

    def get(self,key):
        self.check()
        if key not in self.index:return None
        entry=self.index[key];value=read_checked(self.root,entry['file'],entry['sha256'])
        if value['contract_sha256']!=self.contract_hash or value['key']!=key:
            raise InvalidEvidence('Check record belongs to different run')
        self.events.append(dict(event='reuse',key=key,sha256=entry['sha256']))
        return value['result']

    def put(self,key,result):
        self.check()
        if key in self.index:raise InvalidEvidence('Refusing to overwrite committed check')
        value=dict(key=key,contract_sha256=self.contract_hash,attempt=self.attempt,result=result)
        # Content addressing also preserves orphaned records from interrupted
        # index commits. Recovery never overwrites a previous attempt's evidence.
        file='checks/'+digest(value)+'.json'
        atomic_json(self.root/file,value)
        entry=dict(file=file,sha256=checksum(self.root/file))
        if read_checked(self.root,file,entry['sha256'])!=value:raise OSError('Check read-back differs')
        self.index[key]=entry;atomic_json(self.root/'index.json',self.index)
        self.events.append(dict(event='commit',key=key,sha256=entry['sha256']))
        self.finish('RUNNING')
