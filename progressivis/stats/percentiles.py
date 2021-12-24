from __future__ import annotations

from ..utils.errors import ProgressiveError
from ..core.utils import indices_len, fix_loc
from ..table.module import TableModule, ReturnRunStep
from ..table.table import Table
from ..core.slot import SlotDescriptor
from ..core.decorators import process_slot, run_if_any

import numpy as np

# Should use a Cython implementation eventually
from tdigest import TDigest  # type: ignore


from typing import Optional, List, Union


def _pretty_name(x: float) -> str:
    x *= 100
    if x == int(x):
        return "_%.0f" % x
    else:
        return "_%.1f%%" % x


class Percentiles(TableModule):
    parameters = [
        ("percentiles", np.dtype(np.object_), [0.25, 0.5, 0.75]),
        ("history", np.dtype(int), 3),
    ]
    inputs = [SlotDescriptor("table", type=Table)]

    def __init__(
        self,
        column: str,
        percentiles: Optional[Union[List[float], np.ndarray]] = None,
        **kwds
    ):
        if not column:
            raise ProgressiveError("Need a column name")
        super(Percentiles, self).__init__(**kwds)
        self._columns = [column]
        self.default_step_size = 1000
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
        self.result = Table(
            self.generate_table_name("percentiles"), dshape=dshape, create=True
        )

    def is_ready(self) -> bool:
        if self.get_input_slot("table").created.any():
            return True
        return super(Percentiles, self).is_ready()

    def reset(self) -> None:
        self.tdigest = TDigest()

    @process_slot("table", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            dfslot = ctx.table
            indices = dfslot.created.next(step_size)
            steps = indices_len(indices)
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=steps)
            input_df = dfslot.data()
            x = self.filter_columns(input_df, fix_loc(indices))
            self.tdigest.batch_update(x[0])
            df = self.table
            values = {}
            for n, p in zip(self._pername, self._percentiles):
                values[n] = self.tdigest.percentile(p * 100)
            df.add(values)
            # with self.lock:
            #     df.loc[run_number] = values
            #     if len(df) > self.params.history:
            #         self._df = df.loc[df.index[-self.params.history:]]
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
