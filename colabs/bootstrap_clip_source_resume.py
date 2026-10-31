"""Install issued D1 dependencies before a fresh interpreter imports its runner.

Operational launcher only: the issued source overlay and resume contract stay
unchanged. Never import training libraries into this bootstrap process.
"""
import os
import subprocess
import sys


REQUIREMENTS = ['datasets==2.21.0', 'transformers==4.57.3',
                'numpy==2.1.3', 'pyyaml==6.0.3', 'pytest', 'matplotlib']


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit('Expected issued runner path and its arguments')
    print('D1_BOOTSTRAP: installing dependencies before importing the experiment.', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', *REQUIREMENTS], check=True)
    print('D1_BOOTSTRAP: launching a fresh worker; existing resume identity preserved.', flush=True)
    # exec also keeps the logged child PID: interruption still terminates the
    # actual experiment rather than leaving an orphaned worker holding a lease.
    os.execv(sys.executable, [sys.executable, '-u', *argv])


if __name__ == '__main__':
    main()
