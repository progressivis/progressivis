"""
Jupyter notebook magics for Progressivis
"""

from __future__ import annotations

import yaml
import sys
from . import aio
from IPython.core.magic import (
    Magics,
    magics_class,
    cell_magic,
    line_cell_magic,
    needs_local_scope,
)
from typing import Optional, Any


# https://gist.github.com/nkrumm/2246c7aa54e175964724
@magics_class
class ProgressivisMagic(Magics):
    @line_cell_magic  # type: ignore
    @needs_local_scope  # type: ignore
    def progressivis(
        self, line: str, cell: Optional[str] = None, local_ns: Any = None
    ) -> Any:
        from IPython.display import clear_output

        if cell is None:
            clear_output()  # type: ignore
            for ln in yaml.dump(dict(eval(line, local_ns))).split("\n"):
                print(ln)
            sys.stdout.flush()
        else:
            ps_dict = eval(line, local_ns)
            ps_dict.update(yaml.safe_load(cell))
            return ps_dict

    @cell_magic  # type: ignore
    @needs_local_scope  # type: ignore
    def from_input(
        self, line: str, cell: str, local_ns: Optional[Any] = None
    ) -> aio.Task[Any]:
        module = eval(line, local_ns)
        return aio.create_task(module.from_input(yaml.safe_load(cell)))


def load_ipython_extension(ipython: Any) -> None:
    from IPython import get_ipython  # type: ignore

    ip = get_ipython()  # type: ignore
    ip.register_magics(ProgressivisMagic)
