import numpy as np
from ..core.utils import indices_len, fix_loc, filter_cols
from ..table.module import TableModule
from ..table.table import Table
#from .elementwise import unary_dict_all, binary_dict_all

def make_local(df, px):
    if isinstance(df, dict):
        return make_local_dict(df, px)
    arr = df.to_array()
    result = {}
    for i, n in enumerate(df.columns):
        key = f'{px}.{n}'
        result[key] = arr[:, i]
    return result

def make_local_dict(df, px):
    arr = list(df.values())
    result = {}
    for i, n in enumerate(df.keys()):
        key = f'{px}.{n}'
        result[key] = arr[i]
    return result



def get_ufunc_args(col_expr, local_env):
    assert isinstance(col_expr, tuple) and len(col_expr) in (2, 3)
    if len(col_expr) == 2:
        return col_expr[0], (local_env[col_expr[1]],) # tuple, len==1
    return col_expr[0], (local_env[col_expr[1]], local_env[col_expr[2]])
    
    
class MixUfuncABC(TableModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_expr = self.expr        
    def reset(self):
        if self.result is not None:
            self.result.resize(0)

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
            if not slot.has_buffered() and not isinstance(slot.data(), dict):
                return self._return_run_step(self.state_blocked, steps_run=0)
        if self.result is None:
            if self.has_output_datashape('result'):
                dshape_ = self.get_output_datashape('result')
            else:
                dshape_ = self.get_datashape_from_expr()
                self.ref_expr = {k.split(":")[0]:v for (k, v) in self.expr.items()}
            self.result = Table(self.generate_table_name(f'mix_ufunc'),
                                dshape=dshape_, create=True)
        local_env = {}
        vars_dict = {}
        for n, sl in self._input_slots.items():
            if n == '_params':
                continue
            data_ = sl.data()
            if not isinstance(data_, dict):
                step_size = min(step_size, sl.created.length())
            if (step_size == 0 or data_ is None) and not isinstance(data_, dict):
                return self._return_run_step(self.state_blocked, steps_run=0)
        first_slot = None
        for n, sl in self._input_slots.items():
            if n == '_params':
                continue
            if first_slot is None:
                first_slot = sl
            indices = sl.created.next(step_size)
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
        return self._return_run_step(self.next_state(first_slot), steps_run=steps)
