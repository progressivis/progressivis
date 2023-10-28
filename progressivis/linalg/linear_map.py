from __future__ import annotations

import numpy as np

from ..core.module import ReturnRunStep, def_input, def_output, document
from ..core.utils import indices_len, fix_loc, filter_cols
from ..core.module import Module
from ..table.table import PTable
from ..table.dshape import dshape_projection
from ..core.decorators import process_slot, run_if_any


from typing import Optional, Any, Sequence


@document
@def_input("vectors", type=PTable, hint_type=Sequence[str], doc="Table providing the row vectors")
@def_input(
    "transformation", type=PTable, hint_type=Sequence[str], doc="table providing the transformation matrix"
)
@def_output("result", PTable, doc="result table")
class LinearMap(Module):
    """
    Performs a linear transformation on rows in ``vectors`` table. The ``transformation``
    table (or a subset of its columns) provides the tranformation matrix once all their
    rows are read. Its rows number has to be equal to ``vectors`` columns number.
    """

    def __init__(self, **kwds: Any) -> None:
        """
        Args:
            transf_columns: columns to be included in the transformation matrix. When
                missing all columns are included
        """
        super().__init__(**kwds)
        self._k_dim = 0
        self._transf_columns = None
        self._kwds = {}
        self._transf_cache: Optional[np.ndarray[Any, Any]] = None

    def reset(self) -> None:
        if self.result is not None:
            self.result.resize(0)
        self._transf_cache = None

    @process_slot("vectors", "transformation", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        """
        vectors: (n, k)
        transf:  (k, m)
        result:  (n, m)
        """
        assert self.context
        with self.context as ctx:
            vectors = ctx.vectors.data()
            if vectors is None:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if not self._k_dim:
                self._k_dim = len(ctx.vectors.hint) if ctx.vectors.hint is not None else len(vectors.columns)
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
                self._transf_columns = trans.hint or transformation.columns
                tf = filter_cols(transformation, self._transf_columns)
                self._transf_cache = tf.to_array()
            indices = ctx.vectors.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            vs = self.filter_slot_columns(ctx.vectors, fix_loc(indices))
            array: np.ndarray[Any, Any] = vs.to_array()
            res = np.matmul(array, self._transf_cache)
            if self.result is None:
                dshape_ = dshape_projection(transformation, self._transf_columns)
                self.result = PTable(
                    self.generate_table_name("linear_map"), dshape=dshape_, create=True
                )
            self.result.append(res)
            return self._return_run_step(self.next_state(ctx.vectors), steps_run=steps)
