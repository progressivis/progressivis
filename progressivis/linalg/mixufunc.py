from __future__ import annotations

import numpy as np

from ..core.module import ReturnRunStep
from ..core.utils import fix_loc
from ..table.module import PTableModule
from ..table.table_base import BasePTable
from ..table.table import PTable


from typing import Union, Dict, Any, Tuple, Callable


def make_local(
    df: Union[BasePTable, Dict[str, Any]], px: str
) -> Dict[str, np.ndarray[Any, Any]]:
    if isinstance(df, dict):
        return make_local_dict(df, px)
    arr = df.to_array()
    result: Dict[str, np.ndarray[Any, Any]] = {}
    for i, n in enumerate(df.columns):
        key = f"{px}.{n}"
        result[key] = arr[:, i]
    return result


def make_local_dict(df: Dict[str, Any], px: str) -> Dict[str, np.ndarray[Any, Any]]:
    arr = list(df.values())
    result = {}
    for i, n in enumerate(df.keys()):
        key = f"{px}.{n}"
        result[key] = arr[i]
    return result


def get_ufunc_args(
    col_expr: Any, local_env: Any
) -> Tuple[Callable[..., Any], Tuple[Any, ...]]:
    assert isinstance(col_expr, tuple) and len(col_expr) in (2, 3)
    if len(col_expr) == 2:
        return col_expr[0], (local_env[col_expr[1]],)  # tuple, len==1
    return col_expr[0], (local_env[col_expr[1]], local_env[col_expr[2]])


class MixUfuncABC(PTableModule):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.expr: Dict[str, Any]
        self.ref_expr: Dict[str, Any] = self.expr

    def reset(self) -> None:
        if self.result is not None:
            self.result.resize(0)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        """ """
        reset_all = False
        for slot in self._input_slots.values():
            if slot is None:
                continue
            if slot.updated.any() or slot.deleted.any():
                reset_all = True
                break
        if reset_all:
            for slot in self.input_slot_values():
                slot.reset()
                slot.update(run_number)
            self.reset()
        for slot in self._input_slots.values():
            if slot is None:
                continue
            if not slot.has_buffered() and not isinstance(slot.data(), dict):
                return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            if self.has_output_datashape("result"):
                dshape_ = self.get_output_datashape("result")
            else:
                dshape_ = self.get_datashape_from_expr()
                self.ref_expr = {k.split(":")[0]: v for (k, v) in self.expr.items()}
            self.result = PTable(
                self.generate_table_name("mix_ufunc"), dshape=dshape_, create=True
            )
        local_env = {}
        # vars_dict = {}
        for sl in self.input_slot_values():
            n = sl.input_name
            if n == "_params":
                continue
            data_ = sl.data()
            if not isinstance(data_, dict):
                step_size = min(step_size, sl.created.length())
            if (step_size == 0 or data_ is None) and not isinstance(data_, dict):
                return self._return_run_step(self.state_blocked, steps_run=0)
        first_slot = None
        for sl in self.input_slot_values():
            n = sl.input_name
            assert n is not None
            if n == "_params":
                continue
            if first_slot is None:
                first_slot = sl
            indices = sl.created.next(length=step_size)
            data_ = sl.data()
            is_dict = isinstance(data_, dict)
            df = data_ if is_dict else self.filter_columns(data_, fix_loc(indices), n)
            if is_dict:
                sl.clear_buffers()
            dict_ = make_local(df, n)
            local_env.update(dict_)
        result = {}
        steps = None
        for c in self.result.columns:
            col_expr_ = self.ref_expr[c]
            ufunc, args = get_ufunc_args(col_expr_, local_env)
            result[c] = ufunc(*args)
            if steps is None:
                steps = len(result[c])
        self.result.append(result)
        assert steps is not None and first_slot is not None
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
