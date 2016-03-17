from progressivis.core.utils import ProgressiveError, indices_len
from progressivis.core.dataframe import DataFrameModule
from progressivis.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

class KMeans(DataFrameModule):
    def __init__(self, k, columns=None, **kwds):
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Percentiles, self).__init__(dataframe_slot='percentiles', **kwds)
        self._columns = columns
        self._k = k
        self.default_step_size = 1000

    def is_ready(self):
        if self.get_input_slot('df').has_created():
            return True
        return super(Percentiles, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        if dfslot.has_updated() or dfslot.has_deleted():
            dfslot.reset()
            dfslot.update(run_number)
            self.tdigest = TDigest() # reset

        indices = dfslot.next_created(step_size)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=steps)
        input_df = dfslot.data()
        with dfslot.lock:
            if isinstance(indices, slice): # semantic of slice with .loc
                x = input_df.loc[indices.start:indices.stop-1,self._column]
            else:
                x = input_df.loc[indices,self._column]
            self.tdigest.batch_update(x)
        df = self._df
        values = []
        for p in self._percentiles:
            values.append(self.tdigest.percentile(p*100))
        values.append(run_number)
        with self.lock:
            df.loc[run_number] = values
            if len(df) > self.params.history:
                self._df = df.loc[df.index[-self.params.history:]]
        return self._return_run_step(dfslot.next_state(),
                                     steps_run=steps, reads=steps, updates=len(self._df))

