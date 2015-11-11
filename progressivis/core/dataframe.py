from . import Slot, Module, SlotDescriptor

import pandas as pd

from progressivis.core.changemanager import ChangeManager

class DataFrameSlot(Slot):
    def __init__(self, output_module, output_name, input_module, input_name):
        super(DataFrameSlot, self).__init__(output_module, output_name, input_module, input_name)
        self.changes = None

    def update(self, run_number, df=None):
        if df is None:
            df = self.data()

        return self.changemanager.update(run_number, df)

    def reset(self):
        if self.changes is not None:
            self.changes.reset()

    def next_created(self, n=None):
        return self.changes.next_created(n)

    def has_created(self):
        if self.changes:
            return self.changes.has_created()
        return False

    def created_length(self):
        return self.changes.created_length()
    
    def next_updated(self, n=None):
        return self.changes.next_updated(n)

    def has_updated(self):
        if self.changes:
            return self.changes.has_updated()
        return False

    def updated_length(self):
        return self.changes.updated_length()

    def next_deleted(self, n=None):
        return self.changes.next_deleted(n)
    
    def has_deleted(self):
        if self.changes:
            return self.changes.has_deleted()
        return False

    def deleted_length(self):
        return self.changes.deleted_length()

    def next_state(self):
        if self.changes:
            return self.changes.next_state()
        else:
            return Module.state_blocked

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
            return []

    @property
    def changemanager(self):
        if self.changes is None:
            self.changes = ChangeManager()
        return self.changes
 

class DataFrameModule(Module):
    def __init__(self, dataframe_slot='df', **kwds):
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor(dataframe_slot, type=pd.DataFrame, required=False)])
        super(DataFrameModule, self).__init__(**kwds)
        self._dataframe_slot = dataframe_slot
        self._df = None
        self._stats = {}

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

    def update_timestamps(self):
        if self._df is not None:
            return self._df[Module.UPDATE_COLUMN]
        return Module.EMPTY_COLUMN

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
        self._df[self.UPDATE_COLUMN] = 0

    def predict_step_size(self, duration):
        return 1
    
    def run_step(self,run_number,step_size,howlong):
        raise StopIteration()
