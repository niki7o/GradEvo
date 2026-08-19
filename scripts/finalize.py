from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / 'notebooks' / 'gradevo_analysis.ipynb'
PAPER_TYP = REPO_ROOT / 'paper' / 'gradevo.typ'

def _sh(cmd: list[str], cwd: Path | None=None) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or REPO_ROOT)
    if result.returncode != 0:
        print(f'!! command failed with exit {result.returncode}', file=sys.stderr)
        sys.exit(result.returncode)

def main() -> None:
    py = sys.executable
    _sh([py, 'scripts/build_results_snapshot.py'])
    _sh([py, 'scripts/build_notebook.py'])
    jupyter = shutil.which('jupyter') or str(Path(py).with_name('jupyter'))
    _sh([jupyter, 'nbconvert', '--to', 'notebook', '--execute', '--inplace', str(NOTEBOOK)])
    typst = shutil.which('typst')
    if typst is None:
        print('!! typst not on PATH; skipping paper compile', file=sys.stderr)
        return
    _sh([typst, 'compile', str(PAPER_TYP)])
    print('\nFinalization complete.')
    print(f'  Notebook: {NOTEBOOK}')
    print(f"  Paper:    {PAPER_TYP.with_suffix('.pdf')}")
if __name__ == '__main__':
    main()
