from progressive.core.common import ProgressiveError
from progressive.core.utils import typed_dataframe
from progressive.core.dataframe import DataFrameModule
from progressive.core.slot import SlotDescriptor

import numpy as np
import pandas as pd

from tdigest import TDigest

def _pretty_name(x):
    x *= 100
    if x == int(x):
        return '%.0f%%' % x
    else:
        return '%.1f%%' % x

class Percentiles(DataFrameModule):
    parameters = [('percentiles', object, [0.25, 0.5, 0.75])]
                  
    def __init__(self, column, percentiles=None, **kwds):
        if not column:
            raise ProgressiveError('Need a column name')
        self._add_slots(kwds,'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame)])
        super(Percentiles, self).__init__(dataframe_slot='percentiles', **kwds)
        self._column = column
        self.default_step_size = 1000
        self.tdigest = TDigest()

        if percentiles is None:
            percentiles = np.array([0.25, 0.5, 0.75])
        else:
            # get them all to be in [0, 1]
            percentiles = np.asarray(percentiles)
            if (percentiles > 1).any():
                percentiles = percentiles / 100.0
                msg = ("percentiles should all be in the interval [0, 1]. "
                       "Try {0} instead.")
                raise ValueError(msg.format(list(percentiles)))
            if (percentiles != 0.5).all():  # median isn't included
                lh = percentiles[percentiles < .5]
                uh = percentiles[percentiles > .5]
                percentiles = np.hstack([lh, 0.5, uh])

        self._percentiles = percentiles
        #self._df = pd.DataFrame({'description': [np.nan], self.UPDATE_COLUMN: [self.EMPTY_TIMESTAMP]}, index=index)
        columns = [_pretty_name(x) for x in self._percentiles] + [self.UPDATE_COLUMN]
        dtypes = [np.dtype(float)] * len(columns)
        values = [np.nan] * len(columns)
        self._df = typed_dataframe(columns, dtypes, values)

    def is_ready(self):
        if not self.get_input_slot('df').is_buffer_empty():
            return True
        return super(Percentiles, self).is_ready()

    def run_step(self,run_number,step_size,howlong):
        dfslot = self.get_input_slot('df')
        input_df = dfslot.data()
        dfslot.update(self._start_time, input_df)
        if len(dfslot.deleted) or len(dfslot.updated) > len(dfslot.created):
            # should restart from time 0
            raise ProgressiveError('Percentile module does not manage updates or deletes')
        dfslot.buffer_created()

        indices = dfslot.next_buffered(step_size)
        steps = len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=steps)
        x = input_df.loc[indices, self._column]
        self.tdigest.batch_update(x)
        df = self._df
        values = []
        for p in self._percentiles:
            values.append(self.tdigest.percentile(p*100))
        values.append(np.nan)
        df.loc[0] = values
        return self._return_run_step(self.state_ready, steps_run=steps, reads=steps, updates=len(self._df))

