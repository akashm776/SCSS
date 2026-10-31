"""Latest-only operational launcher around the exact issued D1 source payload."""
import ast
import base64
import hashlib
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]
ISSUED_SHA = 'c343fa83103113fe2ad5e16524f086a773c653a048ccd4ee2112082a790f468c'


def build():
    cell = (ROOT/'colabs/clip_source_resume_bootstrap_drive_one_cell.py').read_text()
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(cell).body
              if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
              and n.targets[0].id in ['PAYLOAD', 'BUNDLE_SHA256']}
    if values['BUNDLE_SHA256'] != ISSUED_SHA or hashlib.sha256(
            zlib.decompress(base64.b64decode(values['PAYLOAD']))).hexdigest() != ISSUED_SHA:
        raise ValueError('Wrong issued payload')
    marker = "    print('FULL_SUBPROCESS_LOG:',log,flush=True)"
    additions = ''
    for name in ['run_clip_source_latest.py', 'clip_source_latest_storage.py']:
        source = (ROOT/'colabs'/name).read_text()
        additions += f"    (staging/{name!r}).write_text({source!r},encoding='utf-8')\n"
    old_call = "'-u',str(bootstrap),str(repo/'colabs/run_clip_source_resume.py'),"
    if cell.count(marker) != 1 or cell.count(old_call) != 1:
        raise ValueError('Unexpected previous cell structure')
    cell = cell.replace(marker, additions + marker)
    cell = cell.replace(old_call,
        "'-u',str(bootstrap),str(staging/'run_clip_source_latest.py'),str(repo/'colabs/run_clip_source_resume.py'),")
    cell = cell.replace('RESUME_RUN = None', 'CLEANUP_ONLY = False  # Set True to reclaim space and stop without training.\nRESUME_RUN = None')
    cell = cell.replace("'--bundle-id',BUNDLE_SHA256,", "'--bundle-id',BUNDLE_SHA256,\n                           *(['--cleanup-only'] if CLEANUP_ONLY else []),")
    cell = cell.replace("print('DRIVE_SYNC_COMPLETE: D1 results and log saved; Drive now unmounted.')",
        "print('DRIVE_SYNC_COMPLETE: latest-only cleanup saved; no training started.' if CLEANUP_ONLY else 'DRIVE_SYNC_COMPLETE: D1 results and log saved; Drive now unmounted.')")
    cell = '# LATEST-ONLY STORAGE: stop the old cell first. Removes older resume blobs; preserves results and latest states.\n' + cell
    target = ROOT/'colabs/clip_source_latest_drive_one_cell.py'
    target.write_text(cell)
    return target


if __name__ == '__main__':
    print(build())
