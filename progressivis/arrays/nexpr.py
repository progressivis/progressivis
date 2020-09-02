import numpy as np
from ..core.utils import indices_len, fix_loc, filter_cols
from ..table.module import TableModule
from ..table.table import Table
import numexpr as ne

def make_local(df, px):
    arr = df.to_array()
    result = {}
    class _Dummy: pass
    dummy = _Dummy()
    for i, n in enumerate(df.columns):
        key = f'_{px}__{i}'
        result[key] = arr[:, i]
        setattr(dummy, n, key)
    return dummy, result

class NumExprABC(TableModule):
    def reset(self):
        if self._table is not None:
            self._table.resize(0)

    def run_step(self, run_number, step_size, howlong):
        """
        """
        reset_all = False
        for slot in self._input_slots.values():
            if slot is None:
                continue
            if slot.updated.any() or slot.deleted.any():
                reset_all = True
                break
        if reset_all:
            for slot in self._input_slots.values():
                slot.reset()
                slot.update(run_number)
            self.reset()
        for slot in self._input_slots.values():
            if slot is None:
                continue
            if not slot.has_buffered():
                return self._return_run_step(self.state_blocked, steps_run=0)
        if self._table is None:
            dshape_ = self.get_output_datashape("table")
            self._table = Table(self.generate_table_name(f'num_expr'),
                                dshape=dshape_, create=True)
        _expr = self.expr
        local_env = {}
        vars_dict = {}
        for n, sl in self._input_slots.items():
            if n == '_params':
                continue
            step_size = min(step_size, sl.created.length())
            data_ = sl.data()
            if step_size == 0 or data_ is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
        first_slot = None
        for n, sl in self._input_slots.items():
            if n == '_params':
                continue
            if first_slot is None:
                first_slot = sl
            indices = sl.created.next(step_size)
            df = self.filter_columns(sl.data(), fix_loc(indices), n)
            fobj, dict_ = make_local(df, n)
            local_env.update(dict_)
            vars_dict[n] = fobj
        result = {}
        steps = None
        for c in self._table.columns:
            expr_ = self.expr[c]

            expr_ = expr_.format(**vars_dict)
            result[c] = ne.evaluate(expr_, local_dict=local_env)
            if steps is None:
                steps = len(result[c])
        self._table.append(result)
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
