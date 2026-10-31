"""Wrap the already-issued D1 payload; never rebuild its scientific sources."""
import ast
import base64
import hashlib
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ISSUED_SHA = 'c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c'


def build():
    cell = (ROOT/'colabs/clip_source_resume_drive_one_cell.py').read_text()
    values = {n.targets[0].id: ast.literal_eval(n.value)
              for n in ast.parse(cell).body if isinstance(n, ast.Assign)
              and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256']}
    raw = zlib.decompress(base64.b64decode(values['PAYLOAD']))
    if values['BUNDLE_SHA256'] != ISSUED_SHA or hashlib.sha256(raw).hexdigest() != ISSUED_SHA:
        raise RuntimeError('Expected the issued resume bundle; do not silently rebuild it')
    bootstrap = (ROOT/'colabs/bootstrap_clip_source_resume.py').read_text()
    marker = "    print('FULL_SUBPROCESS_LOG:',log,flush=True)"
    replacement = ("    bootstrap = staging/'bootstrap_clip_source_resume.py'\n"
                   f"    bootstrap.write_text({bootstrap!r},encoding='utf-8')\n" + marker)
    call = "logger.run_logged([sys.executable,'-u',str(repo/'colabs/run_clip_source_resume.py'),"
    if cell.count(marker) != 1 or cell.count(call) != 1:
        raise RuntimeError('Issued delivery layout changed')
    cell = cell.replace(marker, replacement).replace(call,
        "logger.run_logged([sys.executable,'-u',str(bootstrap),str(repo/'colabs/run_clip_source_resume.py'),")
    cell = '# Bootstrap repair: install before model imports; original bundle identity retained.\n' + cell
    target = ROOT/'colabs/clip_source_resume_bootstrap_drive_one_cell.py'
    target.write_text(cell)
    return target


if __name__ == '__main__':
    print(build())
