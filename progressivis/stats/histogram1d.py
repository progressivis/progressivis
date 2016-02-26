from progressivis.core.utils import indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)

class Histogram1D(DataFrameModule):
    """
    """
    parameters = [('bins', np.dtype(int), 128),
                  ('delta', np.dtype(float), -5)] # 5%

    schema = [('array', np.dtype(object), None),
              ('min', np.dtype(float), np.nan),
              ('max', np.dtype(float), np.nan),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, column, **kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True),
                         SlotDescriptor('min', type=pd.DataFrame, required=True),
                         SlotDescriptor('max', type=pd.DataFrame, required=True)])
        super(Histogram1D, self).__init__(dataframe_slot='df', **kwds)
        self.column = column
        self.total_read = 0
        self._histo = None
        self._edges = None
        self._bounds = None
        self._df = self.create_dataframe(Histogram1D.schema)
  
    def is_ready(self):
        if self._bounds and self.get_input_slot('df').has_created():
            return True
        return super(Histogram1D, self).is_ready()
      
  
    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        min_slot = self.get_input_slot('min')
        min_slot.update(run_number)
        max_slot = self.get_input_slot('max')
        max_slot.update(run_number)
  
        if dfslot.has_updated() or dfslot.has_deleted():
            logger.debug('resetting histogram')
            dfslot.reset()
            self._histo = None
            self._edges = None
            dfslot.update(run_number)
  
        if not (dfslot.has_created() and min_slot.has_created() and max_slot.has_created()):
            logger.info('input buffers empty')
            return self._return_run_step(self.state_blocked, steps_run=0)
  
        bounds = self.get_bounds(min_slot, max_slot)
        if bounds is None:
            logger.debug('No bounds yet at run %d', run_number)
            return self._return_run_step(self.state_blocked, steps_run=0)
  
        bound_min, bound_max = bounds
        if self._bounds is None:
            delta = self.get_delta(*bounds)
            self._bounds = (bound_min - delta, bound_max + delta)
            logger.info("New bounds at run %d: %s"%(run_number, self._bounds))
        else:
            (old_min, old_max) = self._bounds
            delta = self.get_delta(*bounds)
  
            if(bound_min < old_min or bound_max > old_max) \
              or bound_min > (old_min + delta) or bound_max < (old_max - delta):
                self._bounds = (bound_min - delta, bound_max + delta)
                logger.info('Updated bounds at run %d: %s', run_number, self._bounds)
                dfslot.reset()
                dfslot.update(run_number)
                self._histo = None
  
        (curr_min, curr_max) = self._bounds
        if curr_min >= curr_max:
            logger.error('Invalid bounds: %s', self._bounds)
            return self._return_run_step(self.state_blocked, steps_run=0)
  
        input_df = dfslot.data()
        indices = dfslot.next_created(step_size) # returns a slice or ... ?
        steps = indices_len(indices)
        logger.info('Read %d rows', steps)
        self.total_read += steps
        if isinstance(indices, slice):
            indices = slice(indices.start, indices.stop - 1)
        filtered_df = input_df.loc[indices]
        column = filtered_df[self.column]
        bins = self._edges if self._edges is not None else self.params.bins
        histo = None
        if len(column) > 0:
            histo, self._edges = np.histogram(column, bins=bins, range=[curr_min, curr_max], normed=False)
        if self._histo is None:
            self._histo = histo
        elif histo is not None:
            self._histo += histo
        values = [self._histo, curr_min, curr_max, run_number]
        with self.lock:
            print self._histo
            self._df.loc[run_number] = values
            self._df = self._df.loc[self._df.index[-1:]]
        return self._return_run_step(dfslot.next_state(), steps_run=steps)
  
    def get_bounds(self, min_slot, max_slot):
        min_slot.next_created()
        with min_slot.lock:
            min_df = min_slot.data()
            if len(min_df) == 0 and self._bounds is None:
                return None
            min = self.last_row(min_df)[self.column]
  
        max_slot.next_created() 
        with max_slot.lock:
            max_df = max_slot.data()
            if len(max_df) == 0 and self._bounds is None:
                return None
            max = self.last_row(max_df)[self.column]
  
        return (min, max)

    def get_delta(self, min, max):
        delta = self.params['delta']
        extent = max - min
        if delta < 0:
            return extent*delta/-100.0

