"""Keep the issued payload and latest-only delivery; add lease I/O repair."""
import ast
import base64
import hashlib
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ISSUED_SHA = 'c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c'


def build():
    cell = (ROOT/'colabs/clip_source_latest_drive_one_cell.py').read_text()
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(cell).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256']}
    if values['BUNDLE_SHA256'] != ISSUED_SHA or hashlib.sha256(
            zlib.decompress(base64.b64decode(values['PAYLOAD']))).hexdigest() != ISSUED_SHA:
        raise ValueError('Wrong issued payload')
    marker = "    print('FULL_SUBPROCESS_LOG:',log,flush=True)"
    call = "str(bootstrap),str(staging/'run_clip_source_latest.py'),"
    if cell.count(marker) != 1 or cell.count(call) != 1:
        raise ValueError('Unexpected previous cell structure')
    additions = ''
    for name in ['run_clip_source_lease_repair.py', 'clip_source_lease_repair.py']:
        source = (ROOT/'colabs'/name).read_text()
        additions += f"    (staging/{name!r}).write_text({source!r},encoding='utf-8')\n"
    cell = cell.replace(marker, additions + marker)
    cell = cell.replace(call, "str(bootstrap),str(staging/'run_clip_source_lease_repair.py'),str(staging/'run_clip_source_latest.py'),")
    cell = '# LEASE READ REPAIR: latest-only retention and issued resume identity preserved.\n' + cell
    target = ROOT/'colabs/clip_source_lease_repair_drive_one_cell.py'
    target.write_text(cell)
    return target


if __name__ == '__main__':
    print(build())
