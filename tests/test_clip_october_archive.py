import json
from pathlib import Path

import pytest

from scripts import archive_clip_october_evidence as a


def make_export(tmp_path):
    run, bundle = 'fixture', 'bundle'
    raw = {
        'completion.json': json.dumps({'status':'complete'}).encode(),
        'comparison.json': json.dumps({'loss':0.01}).encode(),
        'training_batches.json': json.dumps({'caption':'not public'}).encode(),
        'features.npz': b'not published',
    }
    marker = json.dumps(dict(run_id=run, status='verified_tree',
        files={n:a.digest(b) for n,b in raw.items()})).encode()
    files = {**raw, 'DRIVE_BACKUP_MANIFEST.json':marker}
    selected = {n:b for n,b in files.items() if a.retain(n,b)}
    audit = dict(run_id=run, bundle_id=bundle,
        files={n:a.digest(b) for n,b in selected.items()}, omitted=sorted(set(files)-set(selected)))
    for name, data in selected.items():
        (tmp_path/name).write_bytes(data)
    a.write_json(tmp_path/'export_audit.json', audit)
    return (run, a.digest(marker)['sha256'], bundle)


@pytest.mark.parametrize('name,value', [
    ('training_batches.json', {'indices':[1]}),
    ('unexpected.json', {'rows':[{'caption':'private'}]}),
    ('unexpected.json', {'captions':['private']}),
    ('unexpected.jsonl', {'texts':['private']}),
    ('checkpoint.pt', {}),
    ('personal.pdf', {}),
])
def test_sensitive_inputs_not_retained(name,value):
    assert not a.retain(name,json.dumps(value).encode())


def test_numeric_metadata_is_retained():
    assert a.retain('features_native.json', b'{"feature_sha256":"hash","rows":512}')
    assert a.retain('comparison.json', b'{"loss":0.1,"validation_caption_count":512}')


def test_export_checks_original_manifest_not_only_editable_audit(tmp_path):
    spec = make_export(tmp_path)
    a.verify_export(tmp_path,spec)
    path = tmp_path/'comparison.json'
    path.write_text('{"loss":0.2}')
    audit = json.loads((tmp_path/'export_audit.json').read_text())
    audit['files']['comparison.json'] = a.digest(path.read_bytes())
    a.write_json(tmp_path/'export_audit.json',audit)
    with pytest.raises(ValueError,match='original manifest'):
        a.verify_export(tmp_path,spec)


def test_missing_file_and_wrong_pin_fail(tmp_path):
    spec = make_export(tmp_path)
    with pytest.raises(ValueError,match='pin'):
        a.verify_export(tmp_path,(spec[0],'wrong',spec[2]))
    (tmp_path/'comparison.json').unlink()
    with pytest.raises(FileNotFoundError):
        a.verify_export(tmp_path,spec)


def test_bad_inventory_fails(tmp_path):
    spec = make_export(tmp_path)
    audit = json.loads((tmp_path/'export_audit.json').read_text())
    audit['omitted'].append('invented.json')
    a.write_json(tmp_path/'export_audit.json',audit)
    with pytest.raises(ValueError,match='inventory'):
        a.verify_export(tmp_path,spec)


def test_unsafe_export_path_rejected():
    with pytest.raises(ValueError,match='Unsafe'):
        a.retain('../escape.json',b'{}')


def test_published_october_archives_and_comparison():
    root = Path(__file__).resolve().parents[1]/'experiment_results'
    for name,spec in a.STUDIES.items():
        a.verify_export(root/name,spec)
    a.verify_derived(root)
