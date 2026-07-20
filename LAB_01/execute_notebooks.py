import json, sys, io, traceback, os, time
from contextlib import redirect_stdout, redirect_stderr

# Use non-interactive backend so plotting does not hang in a headless run
import matplotlib
matplotlib.use('Agg')

def is_magic_or_shell(src):
    """Return True if the cell source is a Jupyter magic/shell command that
    cannot be executed as plain Python."""
    for line in src.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('!') or line.startswith('%'):
            return True
        if 'get_ipython()' in line:
            return True
    return False

def run_notebook(nb_path):
    print(f"\nExecuting {nb_path}...", flush=True)
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)

    def display(obj):
        if hasattr(obj, 'to_string'):
            print(obj.to_string())
        else:
            print(repr(obj))

    ns = {'display': display}

    code_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    for idx, i in enumerate(code_cells, 1):
        cell = nb['cells'][i]
        src = ''.join(cell['source'])
        if is_magic_or_shell(src):
            print(f"  Cell {i} ({idx}/{len(code_cells)}) ... skipping shell/magic cell", flush=True)
            cell['outputs'] = []
            continue
        print(f"  Cell {i} ({idx}/{len(code_cells)}) ...", end='', flush=True)
        start = time.time()
        out = io.StringIO()
        err = io.StringIO()
        cell['outputs'] = []
        try:
            with redirect_stdout(out), redirect_stderr(err):
                exec(src, ns)
            elapsed = time.time() - start
            print(f" done in {elapsed:.1f}s", flush=True)
            stdout = out.getvalue()
            stderr = err.getvalue()
            if stdout:
                cell['outputs'].append({
                    'output_type': 'stream',
                    'name': 'stdout',
                    'text': stdout.splitlines(keepends=True)
                })
            if stderr:
                cell['outputs'].append({
                    'output_type': 'stream',
                    'name': 'stderr',
                    'text': stderr.splitlines(keepends=True)
                })
        except Exception as e:
            elapsed = time.time() - start
            print(f" FAILED after {elapsed:.1f}s", flush=True)
            stderr = err.getvalue()
            tb = traceback.format_exc()
            if stderr:
                cell['outputs'].append({
                    'output_type': 'stream',
                    'name': 'stderr',
                    'text': stderr.splitlines(keepends=True)
                })
            cell['outputs'].append({
                'output_type': 'error',
                'ename': type(e).__name__,
                'evalue': str(e),
                'traceback': tb.splitlines(keepends=True)
            })
            print(f"ERROR in {nb_path}: {type(e).__name__}: {e}", flush=True)
            with open(nb_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            return False

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print(f"  Saved executed notebook: {nb_path}", flush=True)
    return True

notebooks = [
    '23MID0043_Lab01_Ames_Housing.ipynb',
    '23MID0043_Lab01_California.ipynb',
    '23MID0043_Lab01_UCI_RealEstate.ipynb'
]

failed = []
for nb in notebooks:
    ok = run_notebook(nb)
    if not ok:
        failed.append(nb)

if failed:
    print(f"\nFailed notebooks: {failed}", flush=True)
    sys.exit(1)
else:
    print("\nAll notebooks executed successfully.", flush=True)
