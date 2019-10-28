# -*- coding: utf-8 -*-
"""Fast Approximate Reservoir Sampling.
See http://erikerlandson.github.io/blog/2015/11/20/very-fast-reservoir-sampling/
and https://en.wikipedia.org/wiki/Reservoir_sampling

Vitter, Jeffrey S. (1 March 1985). "Random sampling with a reservoir" (PDF). ACM Transactions on Mathematical Software. 11 (1): 37-57. doi:10.1145/3147.3165.
"""
from __future__ import absolute_import, division, print_function

from progressivis import SlotDescriptor
from ..core.bitmap import bitmap
from ..table import Table
from ..table.module import TableModule
from ..core.utils import indices_len
from ..table import TableSelectedView

import numpy as np

import logging
logger = logging.getLogger(__name__)

def has_len(d):
    return hasattr(d, '__len__')

class Sample(TableModule):
    parameters = [('samples',  np.dtype(int), 50)]

    def __init__(self, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('table', type=Table)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('select', type=bitmap, required=False)])
        
        super(Sample, self).__init__(**kwds)
        self._tmp_table = Table(self.generate_table_name('sample'),
                            dshape='{select: int64}',
                            create=True)
        self._size = 0 # holds the size consumed from the input table so far
        self._bitmap = None
        self._table = None
        
    def reset(self):
        self._tmp_table.resize(0)
        self._size = 0
        self._bitmap = None
        self.get_input_slot('table').reset()

    def get_data(self, name):
        if name=='select':
            return self.get_bitmap()
        if self._table is not None:
            self._table.selection = self.get_bitmap()
        return super(Sample,self).get_data(name)

    def get_bitmap(self):
        if self._bitmap is None:
            len_ = len(self._tmp_table['select'])
            # Avoid "ValueError: Iteration of zero-sized operands is not enabled"
            self._bitmap = bitmap(self._tmp_table['select']) if len_ else bitmap()
        return self._bitmap

    async def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('table')
        if self._table is None:
            input_table = self.get_input_slot('table').data()
            self._table = TableSelectedView(input_table, bitmap([]))
        dfslot.update(run_number)
        # do not produce another sample is nothing has changed
        if dfslot.deleted.any():
            self.reset()
            dfslot.update(run_number)
        if not dfslot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)

        indices = dfslot.created.next(step_size, as_slice=False)
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        
        k = int(self.params.samples)
        reservoir = self._tmp_table
        res = reservoir['select']
        size = self._size # cache in local variable
        if size < k:
            logger.info('Filling the reservoir %d/%d', size, k)
            # fill the reservoir array until it contains k elements
            rest = indices.pop(k-size)
            reservoir.append({'select': rest})
            size = len(reservoir)

        if len(indices)==0: # nothing else to do
            self._size = size
            if steps:
                self._bitmap = None
            return self._return_run_step(self.state_blocked, steps_run=steps)

        t = 4 * k
        # Threshold (t) determines when to start fast sampling
        # logic. The optimal value for (t) may vary depending on RNG
        # performance characteristics.
        
        if size < t and len(indices)!=0:
            logger.info('Normal sampling from %d to %d', size, t)
        while size < t and len(indices)!=0:
            # Normal reservoir sampling is fastest up to (t) samples
            j = np.random.randint(size)
            if j < k:
                res[j] = indices.pop()[0]
            size += 1

        if len(indices)==0:
            self._size = size
            if steps:
                self._bitmap = None
            return self._return_run_step(self.state_blocked, steps_run=steps)

        logger.info('Fast sampling with %d indices', len(indices))
        while indices:
            # draw gap size (g) from geometric distribution with probability p = k / size
            p = k / size
            u = np.random.rand()
            g = int(np.floor(np.log(u) / np.log(1-p)))
            # advance over the gap, and assign next element to the reservoir
            if (g+1) < len(indices):
                j = np.random.randint(k)
                res[j] = indices[g]
                indices.pop(g+1)
                size += g+1
            else:
                size += len(indices)
                break

        self._size = size
        if steps:
            self._bitmap = None
        return self._return_run_step(self.state_blocked, steps_run=steps)
