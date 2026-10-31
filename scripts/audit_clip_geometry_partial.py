import ast,base64,gzip,hashlib,json,statistics,zlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='Verify and describe this incomplete D0 download; never claim full-grid completion.')
parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parents[1];run=args.run
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
m=load(run/'run_manifest.json');drive=load(run/'DRIVE_BACKUP_MANIFEST.json');sources=load(run/'source_manifest.json');plan=load(run/'plan.json')
assert all(sha(run/n)==v['sha256'] and (run/n).stat().st_size==v['bytes'] for n,v in drive['files'].items())
assert all(sha(run/'source_overlay'/n)==v for n,v in sources.items())
assert all(sha(root/n)==v for n,v in sources.items())
for name,local in [('protocol.json','configs/clip_geometry_audit.json'),('plan.json','configs/clip_geometry_audit_plan.json'),('freeze_manifest.json','configs/clip_geometry_audit_freeze_manifest.json')]:assert sha(run/name)==sha(root/local)
state=plan['seeds']['271828']
for w in [100,500]:assert load(run/f'seed_271828/start_{w:06d}.json')==state['states'][str(w)]['archived_start']
audit=load(run/'seed_271828/trunk_audit.json')
assert hashlib.sha256(json.dumps(audit,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==state['native_trunk_audit_canonical_sha256']
values=defaultdict(lambda:defaultdict(list)); paired={};seen=set();invariants=0;invalid=0;counts=defaultdict(int)
def add(scope,w,key,v):
 if v is not None:values[(scope,w)][key].extend(v if isinstance(v,list) else [v])
for f in sorted(run.glob('seed_*/*/*.jsonl.gz')):
 a=load(f.with_name(f.name.replace('.jsonl.gz','_audit.json')))
 assert a['preserved'] and a['before_sha256']==a['after_sha256'] and a['optimizer_updates']==a['backward_passes']==0
 assert sha(f)==a['records']['sha256']
 with gzip.open(f,'rt') as h:rows=[json.loads(line) for line in h]
 assert len(rows)==64
 w=a['weight_step'];offset=a['batch_offset'];stream=a['stream_seed'];seed=a['training_seed']
 assert (w,stream,offset) not in seen;seen.add((w,stream,offset));counts[(seed,stream,w)]+=1
 for i,r in enumerate(rows):
  assert r['weight_step']==w and r['query_index']==i and r['batch_offset']==offset and r['source_index']==plan['continuation_batches'][f'{seed}:{stream}'][offset][i]
  assert r['N_positive']+r['N_nonpositive']==63 and r['N_negative']+r['N_zero']==r['N_nonpositive']
  qkey=(stream,offset,i);paired.setdefault(qkey,{})[w]=[r['sizes'][str(k)]['eligible'] for k in range(1,9)]
  scopes=['available_partial']+(['complete_stream_2026100101'] if stream==2026100101 else [])+(['primary_available'] if offset==0 else [])
  for scope in scopes:
   for key,v in [('N_positive',r['N_positive']),('N_nonpositive',r['N_nonpositive']),('zero_nonpositive',r['N_nonpositive']==0),('positive_cosine',r['positive_cosine'])]:add(scope,w,key,v)
   for sign in ['positive','nonpositive']:
    add(scope,w,'species/'+sign,[int(s==r['species_id']) for s,t in zip(r['candidate_species_ids'],r['margins']) if (t>0)==(sign=='positive')])
  for k in range(1,9):
   e=r['sizes'][str(k)]
   for scope in scopes:
    for key,v in [('C_available',r['N_positive']>=k),('I_available',r['N_nonpositive']>=k),('eligible',e['eligible'])]:add(scope,w,f'k{k}/{key}',v)
   for regime,s in e['regimes'].items():
    if not s['feasible']:
     assert s['support'] is None and s['candidate'] is None
     continue
    ids=s['support'];assert len(set(ids))==k and i not in ids
    if regime=='C':assert all(v>0 for v in s['contributor_margins'])
    if regime=='I':assert all(v<=0 for v in s['contributor_margins'])
    c=s['candidate'];invalid+=sum(not c[path]['valid'] for path in ['ideal','mixed'])
    if c['ideal']['valid']:
     if k==1:assert abs(c['ideal']['margin']-c['real']['ideal_margin'])<=1e-5
     if regime=='I' and r['positive_cosine']>=0:
      assert c['I_invariant']=='passed' and c['ideal']['margin']<=c['real']['ideal_margin']+1e-5;invariants+=1
    if k!=8:continue
    for scope in scopes:
     add(scope,w,regime+'/source_margin',s['contributor_margins']);add(scope,w,regime+'/real_margin',c['real']['ideal_margin']);add(scope,w,regime+'/same_species',s['same_species_fraction'])
     for path in ['ideal','mixed']:
      add(scope,w,regime+'/'+path+'/margin',c[path]['margin']);add(scope,w,regime+'/'+path+'/norm',c[path]['norm'])
      add(scope,w,regime+'/'+path+'/positive',None if c[path]['margin'] is None else c[path]['margin']>0)
      add(scope,w,regime+'/'+path+'/near_001',None if c[path]['margin'] is None else abs(c[path]['margin'])<=.01)
   for pair,ov in e['overlaps'].items():
    if ov is None:continue
    if pair=='CI':assert ov['fraction']==0
    if k==8:
     for scope in scopes:
      add(scope,w,pair+'/overlap',ov['fraction']);add(scope,w,pair+'/equal',ov['equal'])
   if k==8:
    for regime,h in r['historical'].items():
     for scope in scopes:add(scope,w,'historical/'+regime+'/equal',h['equal'])
assert all(set(v)=={100,500} for v in paired.values())
for key,p in paired.items():
 stream,offset,i=key
 scopes=['available_partial']+(['complete_stream_2026100101'] if stream==2026100101 else [])+(['primary_available'] if offset==0 else [])
 for k in range(1,9):
  e,l=p[100][k-1],p[500][k-1]
  label='both' if e and l else 'early_only' if e else 'late_only' if l else 'neither'
  for scope in scopes:
   for status in ['both','early_only','late_only','neither']:add(scope,'paired',f'k{k}/'+status,label==status)
def stats(v):
 v=np.array(v,dtype=float);return dict(n=len(v),mean=float(v.mean()),min=float(v.min()),max=float(v.max()),q25=float(np.quantile(v,.25)),median=float(np.median(v)),q75=float(np.quantile(v,.75)))
result=dict(status='partial_download_not_completed_audit',run=m['run_id'],bundle=m['bundle_id'],drive_status=drive['status'],verified_files=len(drive['files']),verified_sources=len(sources),batch_forwards=len(seen),target_forwards=624,query_exposures=len(seen)*64,preparation_updates=500,audit_updates=0,failed_or_invalid_mixtures_observed=invalid,I_invariant_checks=invariants,counts=[dict(seed=s,stream=u,weight=w,batches=n) for (s,u,w),n in sorted(counts.items())],groups=[dict(scope=scope,weight=w,metrics={k:stats(v) for k,v in group.items()}) for (scope,w),group in values.items()],limitations=['One seed only; second stream truncated; calibration population absent.','No full-grid macro means or D1 decision.','Missing completion/final tree verification; cannot infer runtime failure from this snapshot.','No synthetic utility or retrieval outcomes were measured.'])
args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='groups'},indent=2))
for group in result['groups']:
 if group['scope']!='complete_stream_2026100101':continue
 print('COMPLETE FIRST STREAM weight',group['weight'])
 for k,v in group['metrics'].items():
  if k in ['zero_nonpositive','k1/I_available','k2/I_available','k8/I_available','k8/C_available','k8/eligible','k8/both','k8/early_only','k8/late_only','HI/overlap','HC/overlap','species/nonpositive','species/positive'] or k.endswith(('/real_margin','/ideal/margin','/ideal/positive','/source_margin','/same_species')): print(k,v)
