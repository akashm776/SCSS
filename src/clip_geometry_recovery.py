"""Explicit D0 recovery: authenticate prior artifacts before reusing observations."""
import gzip
import json
from pathlib import Path
import shutil

from src import clip_geometry_audit as g
from src import clip_checkpoint_diagnostic as d
from src import clip_optimizer_reset as reset

ORIGINAL_BUNDLE = 'ea3dffeb9c6ab1c139fc6f74fd1d1a6b99d115902d766260650f5760a78a457d'
REFERENCE_SHA256 = '74d7b56f30d8cfb6152612f7f42aabb2c6c6286145cf7985f1d682cbeacddeb3'
ORIGINAL_RUN = 'clip_geometry_audit_20261008T221455_073360Z'


def read(path):
    return json.loads(Path(path).read_text())


def safe_member(root, name):
    relative = Path(name)
    g.require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe recovery member')
    path = root / relative
    g.require(not path.is_symlink() and path.resolve().is_relative_to(root.resolve()), 'Recovery symlink escape')
    return path


def key(context):
    return tuple(context[n] for n in ['training_seed','population','stream_seed','batch_offset','weight_step'])


class Recovery:
    def __init__(self, directories, *, bundle_id, p=None, plan=None):
        self.p = g.load_protocol() if p is None else p
        self.plan = read(d.ROOT/self.p['plan']) if plan is None else plan
        self.entries = {}; self.sources = []; self.used = set(); self.new_forwards = 0
        g.require(bool(directories), 'Explicit recovery source required')
        for directory in directories:
            self._inspect(Path(directory), bundle_id)
        g.require(bool(self.entries), 'No authenticated complete batch shards to recover')

    def _inspect(self, root, bundle_id):
        manifest = read(root/'run_manifest.json')
        inventory = read(root/'DRIVE_BACKUP_MANIFEST.json')
        g.require(manifest['experiment']==self.p['experiment'] and
            manifest['bundle_id'] in [ORIGINAL_BUNDLE,bundle_id] and
            manifest['freeze_manifest_sha256']==g.FREEZE_SHA256, 'Unknown recovery experiment/bundle/freeze')
        if manifest['bundle_id']==ORIGINAL_BUNDLE:
            g.require(manifest['run_id']==ORIGINAL_RUN, 'Original recovery run differs')
        else:
            g.require(manifest.get('recovery_root_run')==ORIGINAL_RUN, 'Recovery ancestry differs')
        g.require(manifest['status'] in ['running','complete_pending_drive_flush'],
            'Prior run records failure; inspect it before recovery')
        g.require(inventory['run_id']==manifest['run_id'], 'Recovery inventory run differs')
        files = inventory['files']
        g.require(not any(Path(n).name.startswith('failure') for n in files), 'Prior recorded failure requires inspection')
        for name, expected in files.items():
            path = safe_member(root,name)
            g.require(path.is_file() and path.stat().st_size==expected['bytes'] and
                d.checksum(path)==expected['sha256'], 'Recovery checksum mismatch: '+name)
        for name in ['run_manifest.json','source_manifest.json','protocol.json','plan.json','freeze_manifest.json']:
            g.require(name in files, 'Uninventoried recovery provenance: '+name)
        g.require(read(root/'protocol.json')==self.p and read(root/'plan.json')==self.plan and
            d.checksum(root/'freeze_manifest.json')==g.FREEZE_SHA256,'Recovery protocol or plan changed')
        sources = read(root/'source_manifest.json')
        # All issued numerical sources are immutable; recovery execution files may only
        # come from this exact recovery bundle. No local source is replaced from Drive.
        reference=d.ROOT/'configs/clip_geometry_recovery_reference.json'
        g.require(d.checksum(reference)==REFERENCE_SHA256,'Recovery reference changed')
        issued = read(reference)['source_files']
        expected_sources = issued if manifest['bundle_id']==ORIGINAL_BUNDLE else read(d.ROOT/'geometry_recovery_source_manifest.json')
        g.require(sources==expected_sources, 'Recovery source inventory differs from issued bundle')
        for name,sha in sources.items():
            rel='source_overlay/'+name
            g.require(rel in files and d.checksum(safe_member(root,rel))==sha and
                d.checksum(d.ROOT/name)==sha, 'Recovery source changed: '+name)
        seeds_checked=set();count=0
        for name in sorted(files):
            if not name.endswith('.jsonl.gz') or not name.startswith('seed_'):continue
            stem=name[:-len('.jsonl.gz')]
            companions=[stem+'_audit.json',stem+'_summary.json.gz']
            if not all(n in files for n in companions):continue # incompletely committed batch; not reused
            audit=read(root/companions[0]);k=key(audit)
            seed,pop,stream,offset,w=k
            g.require(seed in self.p['training_seeds'] and w in self.p['checkpoint_steps'],'Unknown recovery state')
            if pop=='continuation':
                g.require(stream in self.p['stream_seeds'],'Unknown recovery stream')
                batches=self.plan['continuation_batches'][f'{seed}:{stream}'];group=f'stream_{stream}'
            else:
                g.require(pop=='calibration_feasibility' and stream is None,'Unknown recovery population')
                batches=self.plan['calibration_batches'][str(seed)];group='calibration'
            g.require(0<=offset<len(batches) and name==f'seed_{seed}/{group}/w{w}_batch{offset:03d}.jsonl.gz','Recovery path/context mismatch')
            if seed not in seeds_checked:
                ref=self.plan['seeds'][str(seed)]
                for step in self.p['checkpoint_steps']:
                    n=f'seed_{seed}/start_{step:06d}.json'
                    g.require(n in files and read(root/n)==ref['states'][str(step)]['archived_start'],'Recovery native start differs')
                n=f'seed_{seed}/trunk_audit.json'
                g.require(n in files and reset.digest(read(root/n))==ref['native_trunk_audit_canonical_sha256'],'Recovery trunk differs')
                seeds_checked.add(seed)
            identity_name=f'seed_{seed}/{group}/batch_identity.json'
            g.require(identity_name in files,'Missing recovery batch identity')
            identity=read(root/identity_name)
            ref=self.plan['seeds'][str(seed)]
            expected_identity=(ref['calibration_batch_records_canonical_sha256'] if stream is None else
                ref['streams'][str(stream)]['training_batch_records_sha256'])
            g.require(identity['canonical_sha256']==expected_identity and identity['source_indices']==[i for batch in batches for i in batch],'Recovery batch membership differs')
            g.require(audit['preserved'] and audit['before_sha256']==audit['after_sha256'] and
                audit['optimizer_updates']==audit['backward_passes']==0 and audit['queries']==self.p['batch_size'] and
                audit['records']['sha256']==files[name]['sha256'] and audit['records']['path']==Path(name).name,
                'Invalid recovery state/record audit')
            rows=self._read_rows(root/name)
            self._validate_rows(rows,k,batches[offset])
            if k in self.entries:
                g.require(self.entries[k]['sha256']==files[name]['sha256'],'Conflicting recovery shards')
            else:
                self.entries[k]=dict(path=root/name,audit=audit,sha256=files[name]['sha256'],run_id=manifest['run_id'])
            count+=1
        self.sources.append(dict(directory=str(root),run_id=manifest['run_id'],bundle_id=manifest['bundle_id'],
            inventory_sha256=d.checksum(root/'DRIVE_BACKUP_MANIFEST.json'),verified_files=len(files),complete_shards=count))

    @staticmethod
    def _read_rows(path):
        with gzip.open(path,'rt') as handle:rows=[json.loads(line) for line in handle]
        json.dumps(rows,allow_nan=False)
        return rows

    def _validate_rows(self,rows,k,indices):
        g.require(len(rows)==self.p['batch_size'],'Wrong recovery query count')
        for i,row in enumerate(rows):
            g.require(key(row)==k and row['query_index']==i and row['source_index']==indices[i] and
                row['candidate_source_indices']==[v for j,v in enumerate(indices) if j!=i], 'Recovery query identity differs')
            g.require(row['N_positive']+row['N_nonpositive']==len(indices)-1 and
                row['N_negative']+row['N_zero']==row['N_nonpositive'], 'Recovery sign counts differ')
            for entry in row['sizes'].values():
                for regime,s in entry['regimes'].items():
                    if not s['feasible']:
                        g.require(s['support'] is None and s['candidate'] is None,'Recovery missingness differs')
                    else:
                        if regime in ['C','I']:
                            g.require(all((m>0)==(regime=='C') for m in s['contributor_margins']),'Recovery source sign differs')
                        g.assert_candidate_invariants(s['candidate'],self.p)

    def get(self,context,identities):
        k=key(context)
        if k not in self.entries:
            self.new_forwards+=1
            return None
        g.require(k not in self.used,'Recovery batch reused twice')
        entry=self.entries[k]
        g.require(d.checksum(entry['path'])==entry['sha256'],'Recovery shard changed after verification')
        rows=self._read_rows(entry['path'])
        self._validate_rows(rows,k,[r['source_index'] for r in identities])
        g.require(all(all(row[n]==identity[n] for n in ['source_index','caption_index','species_id'])
            for row,identity in zip(rows,identities)),'Rebuilt recovery caption/species differs')
        self.used.add(k)
        return rows,entry['audit']

    def save_provenance(self,directory):
        directory=Path(directory)
        for i,source in enumerate(self.sources):
            target=directory/'recovery_provenance'/f'source_{i}';target.mkdir(parents=True)
            for name in ['run_manifest.json','DRIVE_BACKUP_MANIFEST.json','source_manifest.json']:
                shutil.copyfile(Path(source['directory'])/name,target/name)
        d.write_json(directory/'recovery_sources.json',dict(root_run=ORIGINAL_RUN,sources=self.sources,
            reusable_batch_forwards=len(self.entries),remaining_batch_forwards=self.p['budget']['total_batch_forwards']-len(self.entries),
            source_folders_unchanged=True,no_outcome_selection=True))

    def finish(self,directory,preparation_updates):
        g.require(self.used==set(self.entries),'Some authenticated recovery shards were not consumed')
        g.require(len(self.used)+self.new_forwards==self.p['budget']['total_batch_forwards'],'Recovery budget mismatch')
        d.write_json(Path(directory)/'recovery_completion.json',dict(reused_batch_forwards=len(self.used),
            new_batch_forwards=self.new_forwards,total_batch_forwards=len(self.used)+self.new_forwards,
            current_attempt_native_preparation_updates=preparation_updates,
            previous_preparation_not_counted_as_new_scientific_evidence=True,source_folders_unchanged=True))
