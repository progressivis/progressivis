"""
Random table generator module.
"""
from __future__ import annotations

from collections import OrderedDict
import logging

import numpy as np

from ..core.module import ReturnRunStep, def_output, document
from ..core.utils import integer_types
from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from ..core.module import Module
from ..table.table import PTable
from ..table.constant import ConstDict
from ..utils.psdict import PDict

from typing import Dict, Union, Any, Callable, Sequence

logger = logging.getLogger(__name__)

RAND = np.random.rand


@document
@def_output("result", PTable, doc="result table")
class RandomPTable(Module):
    """
    Random table generator module
    """
    def __init__(
        self,
        columns: Union[int, Sequence[str]],
        rows: int = -1,
        random: Callable[..., np.ndarray[Any, Any]] = RAND,
        dtype: str = "float64",
        throttle: Union[int, np.integer[Any], float, bool] = False,
        **kwds: Any,
    ) -> None:
        """
        Args:
            columns: columns definition:

                * if it is an ``int`` it provides the number (**n**) of columns named ``_1`` ... ``_n``
                * else it provides sequence of column names
            rows: if positive  (= **n**) stops generation after ``n`` rows else unbounded
            random: ``numpy`` random function, default is :func:`numpy.random.rand`
            dtype: ``numpy`` alike data type
            throttle: limit the number of rows to be generated in a step
            kwds: extra keyword args to be passed to the ``Module`` superclass
        """
        super().__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        self.default_step_size = 1000
        if isinstance(columns, integer_types):
            self._columns = ["_%d" % i for i in range(1, columns + 1)]
        elif isinstance(columns, (list, np.ndarray)):
            self._columns = list(columns)
        else:
            raise ProgressiveError("Invalid type for columns")
        self.rows = rows
        self.random = random
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        dshape = ", ".join([f"{col}: {dtype}" for col in self._columns])
        dshape = "{" + dshape + "}"
        table = PTable(self.generate_table_name("table"), dshape=dshape, create=True)
        self.result = table
        self._columns = table.columns

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.result is not None
        if step_size == 0:
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        logger.info("generating %d lines", step_size)
        table = self.result
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self.rows >= 0 and (len(table) + step_size) > self.rows:
            step_size = self.rows - len(table)
            logger.info("truncating to %d lines", step_size)
            if step_size <= 0:
                raise ProgressiveStopIteration

        values: Dict[str, np.ndarray[Any, Any]] = OrderedDict()
        assert self._columns is not None
        for column in self._columns:
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


class RandomDict(ConstDict):
    def __init__(self, columns: int, **kwds: Any) -> None:
        keys = [f"_{i}" for i in range(1, columns + 1)]
        vals = np.random.rand(columns)
        super().__init__(PDict(dict(zip(keys, vals))), **kwds)
