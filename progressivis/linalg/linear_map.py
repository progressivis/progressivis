from __future__ import annotations

import numpy as np

from ..core.utils import indices_len, fix_loc, filter_cols
from ..table.module import TableModule, ReturnRunStep
from ..table.table import Table
from ..table.dshape import dshape_projection
from ..core.decorators import process_slot, run_if_any
from .. import SlotDescriptor


from typing import List, Optional


class LinearMap(TableModule):
    inputs = [
        SlotDescriptor("vectors", type=Table, required=True),
        SlotDescriptor("transformation", type=Table, required=True),
    ]

    def __init__(self, transf_columns: List[str] = None, **kwds):
        super().__init__(**kwds)
        self._k_dim = len(self._columns) if self._columns else None
        self._transf_columns = transf_columns
        self._kwds = {}
        self._transf_cache: Optional[np.ndarray] = None

    def reset(self) -> None:
        if self.result is not None:
            self.table.resize(0)
        self._transf_cache = None

    @process_slot("vectors", "transformation", reset_cb="reset")
    @run_if_any
    def run_step(self,
                 run_number: int,
                 step_size: int,
                 howlong: float) -> ReturnRunStep:
        """
        vectors: (n, k)
        transf:  (k, m)
        result:  (n, m)
        """
        assert self.context
        with self.context as ctx:
            vectors = ctx.vectors.data()
            if not self._k_dim:
                self._k_dim = len(vectors.columns)
            trans = ctx.transformation
            transformation = trans.data()
            trans.clear_buffers()
            if len(transformation) < self._k_dim:
                if trans.output_module.state <= self.state_blocked:
                    return self._return_run_step(self.state_blocked, steps_run=0)
                else:  # transformation.output_module is zombie etc.=> no hope
                    raise ValueError(
                        "vectors size don't match " "the transformation matrix shape"
                    )
            elif len(transformation) > self._k_dim:
                raise ValueError(
                    "vectors size don't match " " the transformation matrix shape (2)"
                )
            # here len(transformation) == self._k_dim
            if self._transf_cache is None:
                tf = filter_cols(transformation, self._transf_columns)
                self._transf_cache = tf.to_array()
            indices = ctx.vectors.created.next(step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            vs = self.filter_columns(vectors, fix_loc(indices))
            array: np.ndarray = vs.to_array()
            res = np.matmul(array, self._transf_cache)
            if self.result is None:
                dshape_ = dshape_projection(transformation, self._transf_columns)
                self.result = Table(
                    self.generate_table_name("linear_map"), dshape=dshape_, create=True
                )
            self.table.append(res)
            return self._return_run_step(self.next_state(ctx.vectors), steps_run=steps)
