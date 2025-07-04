from __future__ import annotations
import numpy as np
import logging
from ..core.module import Module, ReturnRunStep, def_input, def_output
from .api import PTable, PTableSelectedView
from ..core.pintset import PIntSet
from progressivis.table.compute import ColFunc, SingleColFunc, MultiColFunc
from typing import Optional, Any, Callable, Tuple, Sequence

logger = logging.getLogger(__name__)
Shape = Tuple[int, ...]

def multi_col_adapter(to_decorate: Callable[..., Any], col_var_map: dict[str, str]) -> Callable[..., Any]:
    def _wrapper(index: Any, local_dict: dict[str, str]) -> Any:
        """
        """
        kwargs = {}
        for var, col in col_var_map.items():
            
            kwargs[var] = local_dict[col]
        return to_decorate(**kwargs)

    return _wrapper

class Computed:
    def __init__(self, computed: dict[str, ColFunc] | None = None) -> None:
        self.computed: dict[str, ColFunc] = {} if computed is None else computed

    def add_ufunc_column(
        self,
        name: str,
        col: str,
        ufunc: Callable[..., Any],
        dtype: Optional[np.dtype[Any]] = None,
        xshape: Shape = (),
    ) -> None:
        self.computed[name] = SingleColFunc(
            func=ufunc, base=col, dtype=dtype, xshape=xshape
        )

    def add_multi_col_func(
        self,
        name: str,
        cols: list[str],
        func: Callable[..., Any],
        col_var_map: dict[str, str],
        dtype: Optional[np.dtype[Any]] = None,
        xshape: Shape = (),
    ) -> None:
        adapted = multi_col_adapter(func, col_var_map)
        self.computed[name] = MultiColFunc(
            func=adapted, base=list(cols), dtype=dtype, xshape=xshape
        )


@def_input("table", PTable, hint_type=Sequence[str])
@def_output("result", PTableSelectedView)
class Repeater(Module):
    def __init__(self, computed: Computed, **kwds: Any) -> None:
        super().__init__(**kwds)
        self._computed = computed.computed

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        input_slot = self.get_input_slot("table")
        assert input_slot is not None
        steps = 0
        input_table = input_slot.data()
        if input_table is None:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            cols = (
                input_slot.hint or tuple(input_table.columns)
            ) + tuple(self._computed.keys())
            self.result = PTableSelectedView(
                input_table, PIntSet([]), columns=cols, computed=self._computed  # type: ignore
            )
        deleted: Optional[PIntSet] = None
        if input_slot.deleted.any():
            deleted = input_slot.deleted.next(as_slice=False)
            steps = 1
            if deleted:
                self.result.selection -= deleted
        created: Optional[PIntSet] = None
        if input_slot.created.any():
            created = input_slot.created.next(length=step_size, as_slice=False)
            steps += len(created)
            self.result.selection |= created
        updated: Optional[PIntSet] = None
        if input_slot.updated.any():
            # currently updates are ignored
            updated = input_slot.updated.next(length=step_size, as_slice=False)
            steps += len(updated)
        return self._return_run_step(self.next_state(input_slot), steps)
