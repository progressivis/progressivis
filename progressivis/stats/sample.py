# -*- coding: utf-8 -*-
"""Fast Approximate Reservoir Sampling.
See http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
and https://en.wikipedia.org/wiki/Reservoir_sampling

Vitter, Jeffrey S. (1 March 1985). "Random sampling with a reservoir" (PDF). ACM Transactions on Mathematical Software. 11 (1): 37-57. doi:10.1145/3147.3165.
"""
from __future__ import annotations

import logging

import numpy as np

from progressivis import SlotDescriptor
from ..core.bitmap import bitmap
from ..table import Table
from ..table.module import TableModule, ReturnRunStep
from ..core.utils import indices_len
from ..core.decorators import process_slot, run_if_any
from ..table import TableSelectedView

from typing import Optional, Any, cast

logger = logging.getLogger(__name__)


def has_len(d):
    return hasattr(d, "__len__")


class Sample(TableModule):
    parameters = [("samples", np.dtype(int), 50)]
    inputs = [SlotDescriptor("table", type=Table)]
    outputs = [SlotDescriptor("select", type=bitmap, required=False)]

    def __init__(self, **kwds):
        super(Sample, self).__init__(output_required=False, **kwds)
        self._tmp_table = Table(
            self.generate_table_name("sample"), dshape="{select: int64}", create=True
        )
        self._size = 0  # holds the size consumed from the input table so far
        self._bitmap: Optional[bitmap] = None
        self.result: Optional[TableSelectedView] = None

    def reset(self) -> None:
        self._tmp_table.resize(0)
        self._size = 0
        self._bitmap = None
        slot = self.get_input_slot("table")
        if slot is not None:
            slot.reset()

    def get_data(self, name: str) -> Any:
        if name == "select":
            return self.get_bitmap()
        if self.result is not None:
            self.result.selection = self.get_bitmap()
        return super(Sample, self).get_data(name)

    def get_bitmap(self) -> bitmap:
        if self._bitmap is None:
            len_ = len(self._tmp_table["select"])
            # Avoid "ValueError: Iteration of zero-sized operands is not enabled"
            self._bitmap = bitmap(self._tmp_table["select"]) if len_ else bitmap()
        return self._bitmap

    @process_slot("table", reset_if="delete", reset_cb="reset")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            if self.result is None:
                self.result = TableSelectedView(ctx.table.data(), bitmap([]))
            indices = cast(
                bitmap, ctx.table.created.next(length=step_size, as_slice=False)
            )
            steps = indices_len(indices)
            k = int(self.params.samples)
            reservoir = self._tmp_table
            res = reservoir["select"]
            size = self._size  # cache in local variable
            if size < k:
                logger.info("Filling the reservoir %d/%d", size, k)
                # fill the reservoir array until it contains k elements
                rest = indices.pop(k - size)
                reservoir.append({"select": rest})
                size = len(reservoir)

            if len(indices) == 0:  # nothing else to do
                self._size = size
                if steps:
                    self._bitmap = None
                return self._return_run_step(self.state_blocked, steps_run=steps)

            t = 4 * k
            # Threshold (t) determines when to start fast sampling
            # logic. The optimal value for (t) may vary depending on RNG
            # performance characteristics.

            if size < t and len(indices) != 0:
                logger.info("Normal sampling from %d to %d", size, t)
            while size < t and len(indices) != 0:
                # Normal reservoir sampling is fastest up to (t) samples
                j = np.random.randint(size)
                if j < k:
                    res[j] = indices.pop()[0]
                size += 1

            if len(indices) == 0:
                self._size = size
                if steps:
                    self._bitmap = None
                return self._return_run_step(self.state_blocked, steps_run=steps)

            logger.info("Fast sampling with %d indices", len(indices))
            while indices:
                # draw gap size (g) from geometric distribution with probability p = k / size
                p = k / size
                u = np.random.rand()
                g = int(np.floor(np.log(u) / np.log(1 - p)))
                # advance over the gap, and assign next element to the reservoir
                if (g + 1) < len(indices):
                    j = np.random.randint(k)
                    res[j] = indices[g]
                    indices.pop(g + 1)
                    size += g + 1
                else:
                    size += len(indices)
                    break

            self._size = size
            if steps:
                self._bitmap = None
            return self._return_run_step(self.state_blocked, steps_run=steps)
