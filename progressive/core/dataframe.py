from progressive.core.module import *
import pandas as pd

from progressive.core.changemanager import ChangeManager, NIL

class DataFrameSlot(Slot):
    def __init__(self, output_module, output_name, input_module, input_name):
        super(DataFrameSlot, self).__init__(output_module, output_name, input_module, input_name)
        self.changes = None

    def update(self, time, df):
        if self.changes is None:
            self.changes = ChangeManager()
        self.changes.update(time,df)

    def buffer_updated(self):
        if self.changes:
            self.changes.buffer_updated()

    def buffer_created(self):
        if self.changes:
            self.changes.buffer_created()

    def next_buffered(self, n):
        if self.changes:
            return self.changes.next_buffered(n)
        else:
            return NIL

    def is_buffer_empty(self):
        if self.changes:
            return self.changes.is_buffer_empty()
        else:
            return True

    @property
    def last_time(self):
        if self.changes:
            return self.changes.last_time
        else:
            return None

    @property
    def last_index(self):
        if self.changes:
            return self.changes.index
        else:
            return None

    @property
    def updated(self):
        if self.changes:
            return self.changes.updated
        else:
            return None

    @property
    def created(self):
        if self.changes:
            return self.changes.created
        else:
            return None

    @property
    def deleted(self):
        if self.changes:
            return self.changes.deleted
        else:
            return None


class DataFrameModule(Module):
    def __init__(self, dataframe_slot='df', **kwds):
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor(dataframe_slot, type=pd.DataFrame)])
        super(DataFrameModule, self).__init__(**kwds)
        self._dataframe_slot = dataframe_slot
        self._df = None
        self._stats = {}

    def validate_outputs(self):
        self.get_output_slot(self._dataframe_slot)

    def create_slot(self, output_name, input_module, input_name):
        if output_name==self._dataframe_slot:
            return DataFrameSlot(self, output_name, input_module, input_name)
        return super(DataFrameModule, self).create_slot(output_name, input_module, input_name)

    def df(self):
        return self._df

    def get_data(self, name):
        if name==self._dataframe_slot:
            return self.df()
        return super(DataFrameModule, self).get_data(name)

    def set_step_stat(self, name, value):
        self._stats[name] = value

    def add_step_stat(self, name, value):
        if name in self._stats:
            self._stats[name] += value
        else:
            self._stats[name] = value

    def collect_stats(self, ts):
        stats = super(DataFrameModule, self).collect_stats(ts)
        df = self.df()
        if df is not None:
            stats['rows'] = len(df)
            stats['cols'] = len(df.columns)
        if len(self._stats) != 0:
            for k,v in self._stats.iteritems():
                stats[k] = v
            self._stats = {}
        return stats

    def update_timestamps(self):
        if self._df is not None:
            return self._df[Module.UPDATE_COLUMN]
        return EMPTY_COLUMN

    def updated_after(self, run_number=None):
        df = self.df()
        if run_number is None:
            return list(df.index)
        else:
            return list(df.index[df[Module.UPDATE_COLUMN] > run_number])

    def add_timestamp(self, run_number, selection=None):
        if self._df is None:
            return
        if not selection:
            self._df[Module.UPDATE_COLUMN] = run_number
        else:
            self._df.loc[selection,Module.UPDATE_COLUMN] = run_number


class Constant(DataFrameModule):
    def __init__(self, df, **kwds):        
        super(Constant, self).__init__(**kwds)
        self._df = df

    def is_ready(self):
        return True

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        raise StopIteration()
