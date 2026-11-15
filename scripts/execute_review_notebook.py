"""Execute our review notebook in-process, including rich outputs, without ports.

A regular Jupyter 'Run All' also works. This runner is useful in restricted
environments where kernel socket creation is prohibited.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from traitlets.config import Config


def main() -> None:
    """Execute every code cell, save real outputs, and fail on any cell error."""
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    path = root / 'notebooks' / 'review_clustering_hmm.ipynb'
    notebook = nbformat.read(path, as_version=4)
    config = Config()
    config.HistoryManager.hist_file = ':memory:'
    shell = InteractiveShell.instance(config=config)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        started = datetime.now(timezone.utc).isoformat()
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
        cell.execution_count = count
        cell.outputs = []
        if captured.stdout:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            cell.outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for rich in captured.outputs:
            cell.outputs.append(nbformat.v4.new_output('display_data', data=rich.data, metadata=rich.metadata))
        error = result.error_before_exec or result.error_in_exec
        if error:
            cell.outputs.append(nbformat.v4.new_output('error', ename=type(error).__name__, evalue=str(error), traceback=[repr(error)]))
        cell.metadata['execution'] = {'started': started, 'finished': datetime.now(timezone.utc).isoformat()}
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(f'Executed cell {count}: {len(cell.outputs)} outputs', flush=True)
        if error:
            raise RuntimeError(f'Notebook cell {count} failed: {error}') from error
    print(f'Executed {count} code cells successfully: {path}')


if __name__ == '__main__':
    main()
