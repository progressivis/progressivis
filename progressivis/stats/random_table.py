"""
Random table generator module.
"""
from __future__ import annotations

from collections import OrderedDict
import logging

import numpy as np

from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from ..table.module import TableModule, ReturnRunStep
from ..table.table import Table
from ..table.constant import Constant
from ..utils.psdict import PsDict
from ..core.utils import integer_types

from typing import List, Dict, Union, Any

logger = logging.getLogger(__name__)

RAND = np.random.rand


class RandomTable(TableModule):
    "Random table generator module"

    def __init__(
        self,
        columns: Union[int, list, np.ndarray],
        rows=-1,
        random=RAND,
        dtype="float64",
        throttle: Union[int, np.integer, float, bool] = False,
        **kwds: Any,
    ) -> None:
        super(RandomTable, self).__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        self.default_step_size = 1000
        self.columns: Union[List[str], np.ndarray]
        if isinstance(columns, integer_types):
            self.columns = ["_%d" % i for i in range(1, columns + 1)]
        elif isinstance(columns, (list, np.ndarray)):
            self.columns = columns
        else:
            raise ProgressiveError("Invalid type for columns")
        self.rows = rows
        self.random = random
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        dshape = ", ".join([f"{col}: {dtype}" for col in self.columns])
        dshape = "{" + dshape + "}"
        table = Table(self.generate_table_name("table"), dshape=dshape, create=True)
        self.result = table
        self.columns = table.columns

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        if step_size == 0:
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        logger.info("generating %d lines", step_size)
        table = self.table
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self.rows >= 0 and (len(table) + step_size) > self.rows:
            step_size = self.rows - len(table)
            logger.info("truncating to %d lines", step_size)
            if step_size <= 0:
                raise ProgressiveStopIteration

        values: Dict[str, np.ndarray] = OrderedDict()
        for column in self.columns:
            s = self.random(step_size)
            values[column] = s
        table.append(values)
        if len(table) == self.rows:
            next_state = self.state_zombie
        # elif self.throttle:
        #     next_state = self.state_blocked
        else:
            next_state = self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)


class RandomDict(Constant):
    def __init__(self, columns: int, **kwds):
        keys = [f"_{i}" for i in range(1, columns + 1)]
        vals = np.random.rand(columns)
        super().__init__(PsDict(dict(zip(keys, vals))), **kwds)
