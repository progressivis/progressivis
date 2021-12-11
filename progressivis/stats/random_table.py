"""
Random table generator module.
"""

from collections import OrderedDict
import logging

import numpy as np

from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from ..table.module import TableModule
from ..table.table import Table
from ..table.constant import Constant
from ..utils.psdict import PsDict
from ..core.utils import integer_types

logger = logging.getLogger(__name__)

RAND = np.random.rand


class RandomTable(TableModule):
    "Random table generator module"
    def __init__(self, columns, rows=-1, random=RAND, dtype='float64',
                 throttle=False, **kwds):
        super(RandomTable, self).__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        self.default_step_size = 1000
        if isinstance(columns, integer_types):
            self.columns = ["_%d" % i for i in range(1, columns+1)]
        elif isinstance(columns, (list, np.ndarray)):
            self.columns = columns
        else:
            raise ProgressiveError('Invalid type for columns')
        self.rows = rows
        self.random = random
        if throttle and isinstance(throttle, integer_types+(float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        dshape = ", ".join([f"{col}: {dtype}" for col in self.columns])
        dshape = "{" + dshape + "}"
        self.result = Table(self.generate_table_name('table'),
                            dshape=dshape,
                            create=True)
        self.columns = self.result.columns

    def run_step(self, run_number, step_size, howlong):
        if step_size == 0:
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0)
        logger.info('generating %d lines', step_size)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self.rows >= 0 and (len(self.result)+step_size) > self.rows:
            step_size = self.rows - len(self.result)
            logger.info('truncating to %d lines', step_size)
            if step_size <= 0:
                raise ProgressiveStopIteration

        values = OrderedDict()
        for column in self.columns:
            s = self.random(step_size)
            values[column] = s
        self.result.append(values)
        if len(self.result) == self.rows:
            next_state = self.state_zombie
        elif self.throttle:
            next_state = self.state_blocked
        else:
            next_state = self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)


class RandomDict(Constant):
    def __init__(self, columns, **kwds):
        keys = [f'_{i}' for i in range(1, columns+1)]
        vals = np.random.rand(columns)
        super().__init__(PsDict(dict(zip(keys, vals))), **kwds)
