from progressivis.core import ProgressiveError
from progressivis.table.module import TableModule
from progressivis.table.table import Table
from progressivis.core.utils import integer_types

import numpy as np
from collections import OrderedDict
import logging
logger = logging.getLogger(__name__)

rand=np.random.rand


class RandomTable(TableModule):
    def __init__(self, columns, rows=-1, random=rand, throttle=False, **kwds):
        super(RandomTable, self).__init__(**kwds)
        self.default_step_size = 1000
        if isinstance(columns,integer_types):
            self.columns = ["_%d"%i for i in range(1,columns+1)]
        elif isinstance(columns,(list,np.ndarray)):
            self.columns = columns
        else:
            raise ProgressiveError('Invalid type for columns')
        self.rows = rows
        self.random = random
        if throttle and isinstance(throttle, integer_types+(float,)):
            self.throttle = throttle
        else:
            self.throttle = False
        dshape = "{" + (", ".join(["%s: float64"%col for col in self.columns])) + "}"
        self._table = Table(self.generate_table_name('table'),
                            dshape=dshape,
#                            scheduler=self.scheduler(),
                            create=True)
        self.columns = self._table.columns

    def run_step(self,run_number,step_size, howlong):
        if step_size==0: # bug
            logger.error('Received a step_size of 0')
            return self._return_run_step(self.state_ready, steps_run=0, creates=0)
        logger.info('generating %d lines', step_size)
        if self.throttle:
            step_size = np.min([self.throttle,step_size])
        if self.rows >= 0 and (len(self._table)+step_size) > self.rows:
            step_size = self.rows - len(self._table)
            if step_size <= 0:
                raise StopIteration
            logger.info('truncating to %d lines', step_size)

        values = OrderedDict()
        for c in self.columns:
            s = self.random(step_size)
            values[c] = s
        with self.lock:
            self._table.append(values)
        next_state = self.state_blocked if self.throttle else self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)
