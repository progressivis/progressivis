from __future__ import annotations

import numpy as np

from ..core.module import ReturnRunStep
from ..core.utils import fix_loc
from ..core.module import Module
from ..table.table_base import BasePTable
from ..table.table import PTable
import numexpr as ne


from typing import Dict, Any, Tuple, Optional


def _make_local(df: BasePTable, px: str) -> Tuple[Any, Dict[str, np.ndarray[Any, Any]]]:
    arr: np.ndarray[Any, Any] = df.to_array()
    result: Dict[str, np.ndarray[Any, Any]] = {}

    class _Aux:
        pass

    aux = _Aux()
    for i, n in enumerate(df.columns):
        key = f"_{px}__{i}"
        result[key] = arr[:, i]
        setattr(aux, n, key)
    return aux, result


class NumExprABC(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.result: Optional[PTable]  # only for mypy
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
            if not slot.has_buffered():
                return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            if self.has_output_datashape("result"):
                dshape_ = self.get_output_datashape("result")
            else:
                dshape_ = self.get_datashape_from_expr()
                self.ref_expr = {k.split(":")[0]: v for (k, v) in self.expr.items()}
            self.result = PTable(
                self.generate_table_name("num_expr"), dshape=dshape_, create=True
            )
        local_env = {}
        vars_dict = {}
        for sl in self.input_slot_values():
            n = sl.input_name
            if n == "_params":
                continue
            step_size = min(step_size, sl.created.length())
            data_ = sl.data()
            if step_size == 0 or data_ is None:
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
            df = self.filter_columns(sl.data(), fix_loc(indices), n)
            fobj, dict_ = _make_local(df, n)
            local_env.update(dict_)
            vars_dict[n] = fobj
        result = {}
        steps = None
        for c in self.result.columns:
            col_expr_ = self.ref_expr[c]

            col_expr_ = col_expr_.format(**vars_dict)
            result[c] = ne.evaluate(col_expr_, local_dict=local_env)
            if steps is None:
                steps = len(result[c])
        self.result.append(result)
        assert steps is not None and first_slot is not None
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
