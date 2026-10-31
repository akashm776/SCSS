"""Streaming numeric summaries for D0; primary is a view, never extra samples."""
from array import array
from collections import defaultdict
from itertools import combinations
import math
import statistics

import numpy as np
from src.clip_geometry_audit import REGIMES, PAIRS, require


class Distribution:
    def __init__(self):
        self.values=array('d');self.denominator=0;self.exposures=0;self.available_exposures=0
        self.strata=defaultdict(lambda:[0.,0])

    def add(self, values, stratum):
        if not isinstance(values,(list,tuple,np.ndarray)):
            values=[values]
        self.denominator+=len(values);self.exposures+=1
        valid=[float(x) for x in values if x is not None]
        require(all(math.isfinite(x) for x in valid),'Nonfinite summary input')
        self.available_exposures+=bool(valid)
        self.values.extend(valid)
        s=self.strata[stratum];s[0]+=sum(valid);s[1]+=len(valid)

    def finish(self,p,expected):
        v=np.asarray(self.values);n=len(v);edges=p['measurements']['histogram_edges']
        macro=None
        if all(self.strata[s][1] for s in expected):
            byseed=defaultdict(list)
            for s in expected:byseed[s[0]].append(self.strata[s][0]/self.strata[s][1])
            macro=statistics.fmean(statistics.fmean(x) for x in byseed.values())
        return dict(count=n,denominator=self.denominator,missing=self.denominator-n,
            coverage=n/self.denominator if self.denominator else None,
            query_exposures=self.exposures,available_query_exposures=self.available_exposures,
            query_coverage=self.available_exposures/self.exposures if self.exposures else None,
            pooled_mean=float(v.mean()) if n else None,macro_mean=macro,
            minimum=float(v.min()) if n else None,maximum=float(v.max()) if n else None,
            quantiles={str(q):float(np.quantile(v,q,method='linear')) if n else None for q in p['measurements']['quantile_probabilities']},
            histogram=dict(edges=edges,counts=np.histogram(v,bins=edges)[0].tolist(),
                underflow=int((v<edges[0]).sum()),overflow=int((v>edges[-1]).sum()),exact_zero=int((v==0).sum())),
            small_absolute_margin_coverage={str(t):float((np.abs(v)<=t).mean()) if n else None for t in p['measurements']['small_margin_absolute_thresholds']})


def measurements(r):
    yield 'positive_cosine',r['positive_cosine']
    yield 'training_positive_cosine',r['training_positive_cosine']
    for name in ['N_negative','N_zero','N_positive','N_nonpositive','precision_sign_disagreements']:
        yield name,r[name]
    yield 'zero_nonpositive',r['N_nonpositive']==0
    yield 'candidate_margin',r['margins']
    yield 'candidate_training_margin',r['training_margins']
    for label,condition in [('positive',lambda m:m>0),('nonpositive',lambda m:m<=0),('negative',lambda m:m<0),('zero',lambda m:m==0)]:
        yield 'semantics/'+label,[s==r['species_id'] for s,m in zip(r['candidate_species_ids'],r['margins']) if condition(m)]
    for regime,h in r['historical'].items():
        yield f'historical/{regime}/equal',h['equal']
        yield f'historical/{regime}/overlap',h['overlap']
    for k,entry in r['sizes'].items():
        prefix='k'+k+'/'
        yield prefix+'positive_feasible',r['N_positive']>=int(k)
        yield prefix+'nonpositive_feasible',r['N_nonpositive']>=int(k)
        yield prefix+'eligible',entry['eligible']
        for regime,s in entry['regimes'].items():
            pre=prefix+regime+'/'
            yield pre+'feasible',s['feasible']
            yield pre+'source_margin',s['contributor_margins'] if s['feasible'] else [None]*int(k)
            yield pre+'source_training_margin',s['contributor_training_margins'] if s['feasible'] else [None]*int(k)
            yield pre+'rank',s['similarity_ranks'] if s['feasible'] else [None]*int(k)
            yield pre+'same_species',s['same_species_fraction']
            yield pre+'real_same_species',s['real_same_species']
            yield pre+'original_score_support_equal',s['original_score_support_equal']
            c=s['candidate']
            yield pre+'real_margin',c['real']['ideal_margin'] if c else None
            yield pre+'real_training_margin',c['real']['training_margin'] if c else None
            for path in ['ideal','mixed']:
                v=c[path] if c else None
                yield pre+path+'/valid',None if v is None else v['valid']
                for field in ['norm','margin','delta_margin','delta_absolute_margin']:
                    yield pre+path+'/'+field,None if v is None else v[field]
                if path=='mixed':
                    for field in ['training_margin','training_delta_margin','sign_disagreement']:
                        yield pre+path+'/'+field,None if v is None else v[field]
        for a,b in PAIRS:
            pair=entry['overlaps'][a+b];pre=prefix+'paired/'+a+b+'/'
            yield pre+'overlap',None if pair is None else pair['fraction']
            yield pre+'equal',None if pair is None else pair['equal']
            x,y=[entry['regimes'][z] for z in [a,b]]
            same_real=None if pair is None else x['support'][0]==y['support'][0]
            yield pre+'select_equal',same_real
            valid_pair=pair is not None and all(s['candidate'][path]['valid'] for s in [x,y] for path in ['ideal','mixed'])
            yield pre+'exact_candidate_redundancy',pair['equal'] and same_real if valid_pair else None
            for side,s in [(a,x),(b,y)]:
                yield pre+side+'/source_mean',statistics.fmean(s['contributor_margins']) if pair else None
                yield pre+side+'/real_margin',s['candidate']['real']['ideal_margin'] if pair else None
            for path in ['ideal','mixed']:
                common=pair is not None and all(s['candidate'][path]['valid'] for s in [x,y])
                for side,s in [(a,x),(b,y)]:
                    yield pre+path+'/'+side+'/margin',s['candidate'][path]['margin'] if common else None
                yield pre+path+'/margin_difference_B_minus_A',y['candidate'][path]['margin']-x['candidate'][path]['margin'] if common else None


def state_comparisons(early,late):
    require(all(early[k]==late[k] for k in ['population','training_seed','stream_seed','batch_offset','query_index','source_index','caption_index']), 'Unmatched cross-state query')
    require(early['candidate_source_indices']==late['candidate_source_indices'],'Cross-state candidate population differs')
    yield 'positive_cosine_difference',late['positive_cosine']-early['positive_cosine']
    for k,a in early['sizes'].items():
        b=late['sizes'][k];pre='k'+k+'/'
        label=('both' if a['eligible'] and b['eligible'] else 'early_only' if a['eligible'] else 'late_only' if b['eligible'] else 'neither')
        for status in ['both','early_only','late_only','neither']:
            yield pre+'eligibility/'+status,label==status
        for regime in REGIMES:
            x,y=[e['regimes'][regime] for e in [a,b]]
            common=x['feasible'] and y['feasible']
            base=pre+regime+'/'
            yield base+'common_feasible',common
            yield base+'source_mean_difference',statistics.fmean(y['contributor_margins'])-statistics.fmean(x['contributor_margins']) if common else None
            for side,s in [('early',x),('late',y)]:
                yield base+side+'/real_margin',s['candidate']['real']['ideal_margin'] if common else None
                yield base+side+'/source_margin',s['contributor_margins'] if common else [None]*int(k)
            for path in ['ideal','mixed']:
                valid=common and x['candidate'][path]['valid'] and y['candidate'][path]['valid']
                for side,s in [('early',x),('late',y)]:
                    yield base+path+'/'+side+'/margin',s['candidate'][path]['margin'] if valid else None
                yield base+path+'/margin_difference',y['candidate'][path]['margin']-x['candidate'][path]['margin'] if valid else None
        # C-vs-I geometry on the intersection eligible at BOTH states.
        for regime in ['C','I']:
            for side,e in [('early',a),('late',b)]:
                s=e['regimes'][regime]
                yield pre+'common_CI_both_states/'+side+'/'+regime+'/source_margin',s['contributor_margins'] if label=='both' else [None]*int(k)
                yield pre+'common_CI_both_states/'+side+'/'+regime+'/real_margin',s['candidate']['real']['ideal_margin'] if label=='both' else None
            for path in ['ideal','mixed']:
                valid=label=='both' and all(e['regimes'][g]['candidate'][path]['valid'] for e in [a,b] for g in ['C','I'])
                for side,e in [('early',a),('late',b)]:
                    yield pre+'common_CI_both_states/'+path+'/'+side+'/'+regime+'/margin',e['regimes'][regime]['candidate'][path]['margin'] if valid else None


class Accumulator:
    def __init__(self,p,expected):
        self.p=p;self.expected=expected;self.metrics=defaultdict(Distribution);self.queries=0
        self.reasons=defaultdict(int)

    def add(self,record,pairs=None):
        stratum=(record['training_seed'],record['stream_seed'])
        self.queries+=1
        for key,value in (measurements(record) if pairs is None else state_comparisons(record,pairs)):
            self.metrics[key].add(value,stratum)
        if pairs is None:
            for k,e in record['sizes'].items():
                for regime,s in e['regimes'].items():
                    if s['reason']:self.reasons[f'k{k}/{regime}/{s["reason"]}']+=1
                    if s['candidate']:
                        for path in ['ideal','mixed']:
                            reason=s['candidate'][path]['reason']
                            if reason:self.reasons[f'k{k}/{regime}/{path}/{reason}']+=1

    def finish(self):
        return dict(query_exposures=self.queries,missing_reasons=dict(self.reasons),
                    metrics={k:v.finish(self.p,self.expected) for k,v in sorted(self.metrics.items())})


class Summary:
    def __init__(self,p):
        self.p=p;self.groups={};self.record_count=0;self.batch_count=0
        self.passing={str(k):True for k in p['support_sizes'] if k>=2}
        self.zero_nonpositive=False
        self.seen=set()
        self.expected={(pop,seed,stream,offset)
            for seed in p["training_seeds"]
            for pop,streams,offsets in [("continuation",p["stream_seeds"],p["populations"]["extended"]["offsets"]),
                ("calibration_feasibility",[None],p["populations"]["calibration_feasibility"]["batch_offsets"])]
            for stream in streams for offset in offsets}

    def add_batch_pair(self,rows):
        early,late=[rows[w] for w in self.p['checkpoint_steps']]
        require(len(early)==len(late)==self.p['batch_size'],'Wrong matched batch size')
        fields=['population','training_seed','stream_seed','batch_offset']
        key=tuple(early[0][f] for f in fields)
        require(key not in self.seen,'Duplicate audit batch')
        require(key in self.expected,'Unexpected audit batch')
        for weight in self.p['checkpoint_steps']:
            require(all(tuple(r[f] for f in fields)==key and r['weight_step']==weight and r['query_index']==i
                for i,r in enumerate(rows[weight])),'Inconsistent audit batch identities')
        self.seen.add(key)
        self.batch_count+=2;self.record_count+=len(early)+len(late)
        for e,l in zip(early,late):
            for row in [e,l]:
                self.zero_nonpositive |= row['N_nonpositive']==0
                for k in self.passing:
                    self.passing[k] &= all(row['sizes'][k]['regimes'][g]['feasible'] and
                        all(row['sizes'][k]['regimes'][g]['candidate'][path]['valid'] for path in ['ideal','mixed']) for g in ['C','I'])
            scopes=['calibration_feasibility'] if e['population']=='calibration_feasibility' else ['extended']+(['primary'] if e['batch_offset']==0 else [])
            for pop in scopes:
                for weight,row,paired in [(e['weight_step'],e,None),(l['weight_step'],l,None),('paired',e,l)]:
                    for seed,stream in [('all','all'),(e['training_seed'],'all'),(e['training_seed'],e['stream_seed'])]:
                        if e['stream_seed'] is None and stream is None:continue # calibration already has seed aggregate
                        key=(pop,str(weight),str(seed),str(stream))
                        if key not in self.groups:
                            seeds=self.p['training_seeds'] if seed=='all' else [seed]
                            streams=[None] if pop=='calibration_feasibility' else self.p['stream_seeds'] if stream=='all' else [stream]
                            self.groups[key]=Accumulator(self.p,[(s,u) for s in seeds for u in streams])
                        self.groups[key].add(row,paired)

    def finish(self):
        require(self.seen==self.expected,'Incomplete audit membership grid')
        require(self.batch_count==self.p['budget']['total_batch_forwards'] and
                self.record_count==self.p['budget']['total_query_exposures'],'Incomplete audit budget/grid')
        return dict(experiment=self.p['experiment'],batch_forwards=self.batch_count,query_exposures=self.record_count,
            audit_optimizer_updates=0,primary_is_subset=True,d1_protocol_frozen=False,
            all_query_candidate_sizes=[int(k) for k,v in self.passing.items() if v],
            zero_nonpositive_observed=self.zero_nonpositive,
            groups=[dict(population=key[0],weight=key[1],seed=key[2],stream=key[3],**value.finish()) for key,value in sorted(self.groups.items())],
            interpretation='Feasibility/separability only. Conditional distributions are pooled query exposures or labelled source members. '
                'Macro means require every expected stratum. No k or arm chosen, no training-utility result. '
                'Primary is a subset of extended. Three reused seeds, not independent query replications.')


def markdown_report(summary):
    def number(x):
        return 'undefined' if x is None else f'{x:.6g}'

    def cell(metrics,key,median=False):
        value=metrics[key]
        statistic=value['quantiles']['0.5'] if median else value['pooled_mean']
        return f"{number(statistic)} ({value['count']}/{value['denominator']})"

    lines=['# D0 geometry and feasibility audit','',summary['interpretation'],'',
        f"Audit: {summary['batch_forwards']} fixed-state batch forwards; {summary['query_exposures']} query exposures; zero audit updates.",'',
        'All-query C/I sizes passing structural and both-path numerical feasibility: '+str(summary['all_query_candidate_sizes'])+'.',
        'D1 remains unfrozen. Checkpoint preparation is recorded separately in completion.json.','',
        'Tables show pooled statistics and available/total denominators. Source margins pool source members; '
        'real/synthetic margins, species fractions and overlaps use query exposures. Undefined entries are never zero. '
        'Complete all-k, seed, stream and common-population distributions and macro means are in summary.json; '
        'per-batch summaries accompany each query shard. k=8 is a descriptive view, not a selected D1 design.', '']
    for group in summary['groups']:
        if group['seed']!='all' or group['stream']!='all':continue
        m=group['metrics']
        lines += [f"## {group['population']}: weights {group['weight']}",'']
        if group['weight']=='paired':
            lines += ['Eligibility transitions on matched query exposures; entries are fractions (available/total).','',
                '| k | Both | Early only | Late only | Neither |','|---:|---:|---:|---:|---:|']
            for k in range(1,9):
                lines.append('| '+str(k)+' | '+' | '.join(cell(m,f'k{k}/eligibility/{v}') for v in ['both','early_only','late_only','neither'])+' |')
            lines.append('')
            continue
        lines += ['### Feasibility','',f"Zero nonpositive candidates: {cell(m,'zero_nonpositive')}.",'',
            '| k | Positive availability | Nonpositive availability | Common C/I eligibility |','|---:|---:|---:|---:|']
        for k in range(1,9):
            lines.append('| '+str(k)+' | '+' | '.join(cell(m,f'k{k}/{v}') for v in ['positive_feasible','nonpositive_feasible','eligible'])+' |')
        lines += ['','### Geometry at k=8','',
            'Each column uses its own available population. Paired common-population comparisons are recorded separately in summary.json.','',
            '| Regime | Source median margin | Select median | Ideal synth median | Mixed synth median |',
            '|---|---:|---:|---:|---:|']
        for r in REGIMES:
            lines.append('| '+r+' | '+' | '.join(cell(m,f'k8/{r}/{v}',True) for v in ['source_margin','real_margin','ideal/margin','mixed/margin'])+' |')
        lines += ['','### Separability at k=8','',
            '| Pair | Mean overlap | Equal support rate | Equal Select rate | Exact candidate redundancy |',
            '|---|---:|---:|---:|---:|']
        for a,b in PAIRS:
            lines.append('| '+a+b+' | '+' | '.join(cell(m,f'k8/paired/{a+b}/{v}') for v in ['overlap','equal','select_equal','exact_candidate_redundancy'])+' |')
        lines += ['', 'Exact redundancy additionally requires equal selected real IDs and both valid synthesis paths; '
            'equal support sets alone need not give equal Select examples. Historical same-forward support equality: '+
            ', '.join(r+': '+cell(m,f'historical/{r}/equal') for r in ['H','M'])+'.','',
            '### Semantics','', 'Same-species associations are descriptive, not verified false-negative labels.','',
            '| Population | Same-species fraction |','|---|---:|']
        for label in ['positive','nonpositive','negative','zero']:
            lines.append('| Candidate members: '+label+' | '+cell(m,'semantics/'+label)+' |')
        for r in REGIMES:
            lines.append('| '+r+' support (k=8) | '+cell(m,f'k8/{r}/same_species')+' |')
            lines.append('| '+r+' Select (k=8) | '+cell(m,f'k8/{r}/real_same_species')+' |')
        lines.append('')
    return '\n'.join(lines)+'\n'
