"""Install the lease I/O repair before the issued latest-only launcher."""
import importlib.util
import json
import os
from pathlib import Path
import runpy
import sys
import hashlib


def main():
    # Arguments: latest-only launcher, issued runner, then original CLI flags.
    launcher, runner = map(Path, sys.argv[1:3])
    repo = runner.resolve().parents[1]
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    sources = json.loads((repo/'source_resume_source_manifest.json').read_text())
    for name, digest in sources.items():
        if hashlib.sha256((repo/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Issued source overlay changed: ' + name)
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('d1_lease_repair',
        Path(__file__).with_name('clip_source_lease_repair.py'))
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    repair.install()
    print('D1_LEASE_REPAIR: serialized heartbeat/checks; bounded read retries; ownership guard retained.', flush=True)
    sys.argv = [str(launcher), str(runner), *sys.argv[3:]]
    runpy.run_path(str(launcher), run_name='__main__')


if __name__ == '__main__':
    main()
