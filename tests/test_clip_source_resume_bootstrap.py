import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import pytest

from colabs import bootstrap_clip_source_resume as bootstrap
from scripts.build_clip_source_resume_bootstrap_cell import ROOT, ISSUED_SHA


def test_install_failure_never_launches_worker(monkeypatch):
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])
    monkeypatch.setattr(bootstrap.subprocess, 'run', fail)
    monkeypatch.setattr(bootstrap.os, 'execv', lambda *args: pytest.fail('Worker launched'))
    with pytest.raises(subprocess.CalledProcessError):
        bootstrap.main(['runner.py', '--bundle-id', ISSUED_SHA])


def test_fresh_worker_after_install_and_exit_propagation(tmp_path):
    # A fake pip replaces a dependency. The parent deliberately holds its old
    # version in memory. No network, package installation, or GPU is involved.
    (tmp_path/'fake_dependency.py').write_text("VERSION = 'old'\n")
    (tmp_path/'pip.py').write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_name('fake_dependency.py').write_text(\"VERSION = 'new-installed'\\n\")\n")
    worker = tmp_path/'worker.py'
    worker.write_text("import fake_dependency, sys\n"
                      "print('WORKER_VERSION:', fake_dependency.VERSION, flush=True)\n"
                      "assert sys.argv[1:] == ['--bundle-id', 'unchanged']\n"
                      "sys.exit(17)\n")
    command = ("import fake_dependency, runpy, sys; "
               "assert fake_dependency.VERSION == 'old'; "
               "sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")
    result = subprocess.run([sys.executable, '-B', '-c', command,
                             str(ROOT/'colabs/bootstrap_clip_source_resume.py'), str(worker),
                             '--bundle-id', 'unchanged'], cwd=tmp_path,
                            env={**os.environ, 'PYTHONPATH': str(tmp_path), 'PYTHONDONTWRITEBYTECODE': '1'},
                            capture_output=True, text=True)
    assert result.returncode == 17, result.stdout + result.stderr
    assert 'WORKER_VERSION: new-installed' in result.stdout


def test_repaired_delivery_preserves_issued_sources_and_bootstrap():
    def values(path):
        return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256', 'RESUME_RUN']}
    old = values(ROOT/'colabs/clip_source_resume_drive_one_cell.py')
    path = ROOT/'colabs/clip_source_resume_bootstrap_drive_one_cell.py'
    assert values(path) == old
    raw = zlib.decompress(base64.b64decode(old['PAYLOAD']))
    assert hashlib.sha256(raw).hexdigest() == ISSUED_SHA
    sources = json.loads(raw)
    manifest = json.loads(sources.pop('source_resume_source_manifest.json'))
    assert len(manifest) == 135
    for name, source in sources.items():
        assert hashlib.sha256(source.encode()).hexdigest() == manifest[name]
        assert (ROOT/name).read_text() == source
    tree = ast.parse(path.read_text())
    writes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text'
              and isinstance(node.func.value, ast.Name) and node.func.value.id == 'bootstrap']
    assert len(writes) == 1
    assert ast.literal_eval(writes[0].args[0]) == (ROOT/'colabs/bootstrap_clip_source_resume.py').read_text()
    assert "'-u',str(bootstrap),str(repo/'colabs/run_clip_source_resume.py')" in path.read_text()
