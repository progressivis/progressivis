from __future__ import annotations

from progressivis.core.module import (
    Module,
    ReturnRunStep,
    def_input,
    def_output,
    def_parameter,
)
from ..core.utils import indices_len, fix_loc
from ..table.table import PTable
from ..core.decorators import process_slot, run_if_any

import numpy as np

# Should use a Cython implementation eventually


from typing import Optional, List, Union, Any, Sequence


def _pretty_name(x: float) -> str:
    x *= 100
    if x == int(x):
        return "_%.0f" % x
    else:
        return "_%.1f%%" % x


@def_parameter("percentiles", np.dtype(np.object_), [0.25, 0.5, 0.75])
@def_parameter("history", np.dtype(int), 3)
@def_input("table", PTable, hint_type=Sequence[str])
@def_output("result", PTable)
class Percentiles(Module):
    """ """

    def __init__(
        self,
        percentiles: Optional[Union[List[float], np.ndarray[Any, Any]]] = None,
        **kwds: Any,
    ) -> None:
        super(Percentiles, self).__init__(**kwds)
        self.default_step_size = 1000
        from tdigest import TDigest  # type: ignore
        self.tdigest = TDigest()

        if percentiles is None:
            percentiles = np.array([0.25, 0.5, 0.75])
        else:
            # get them all to be in [0, 1]
            percentiles = np.asarray(percentiles)
            if (percentiles > 1).any():
                percentiles = percentiles / 100.0
                msg = (
                    "percentiles should all be in the interval [0, 1]. "
                    "Try {0} instead."
                )
                raise ValueError(msg.format(list(percentiles)))
            if (percentiles != 0.5).all():  # median isn't included
                lh = percentiles[percentiles < 0.5]
                uh = percentiles[percentiles > 0.5]
                percentiles = np.hstack([lh, 0.5, uh])

        self._percentiles = percentiles
        self._pername: List[str] = [_pretty_name(x) for x in self._percentiles]
        dshape = "{" + ",".join(["%s: real" % n for n in self._pername]) + "}"
        self.result = PTable(
            self.generate_table_name("percentiles"), dshape=dshape, create=True
        )

    def is_ready(self) -> bool:
        slot = self.get_input_slot("table")
        if slot is not None and slot.created.any():
            return True
        return super(Percentiles, self).is_ready()

    def reset(self) -> None:
        from tdigest import TDigest
        self.tdigest = TDigest()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        assert self.result is not None
        with self.context as ctx:
            dfslot = ctx.table
            assert dfslot.hint is not None
            assert len(dfslot.hint) == 1
            indices = dfslot.created.next(length=step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=steps)
            x = self.filter_slot_columns(ctx.table, fix_loc(indices))
            self.tdigest.batch_update(x[0])
            df = self.result
            values = {}
            for n, p in zip(self._pername, self._percentiles):
                values[n] = self.tdigest.percentile(p * 100)
            df.add(values)
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
