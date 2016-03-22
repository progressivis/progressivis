from . import Slot, Module, SlotDescriptor

import pandas as pd

from progressivis.core.changemanager import ChangeManager

class DataFrameSlot(Slot):
    def __init__(self, output_module, output_name, input_module, input_name):
        super(DataFrameSlot, self).__init__(output_module, output_name, input_module, input_name)
        self.changes = None

    def update(self, run_number):
        c = self.changemanager
        with self.lock:
            df = self.data()
            return c.update(run_number,df)

    def reset(self):
        if self.changes is not None:
            self.changes.reset()

    def flush_buffers(self):
        return self.changes.flush_buffers()

    def flush_created(self):
        return self.changes.flush_created()

    def next_created(self, n=None):
        return self.changes.next_created(n)

    def has_created(self):
        if self.changes:
            return self.changes.has_created()
        return False

    def created_length(self):
        if self.changes:
            return self.changes.created_length()
        return 0
    
    def flush_updated(self):
        return self.changes.flush_updated()

    def next_updated(self, n=None):
        return self.changes.next_updated(n)

    def has_updated(self):
        if self.changes:
            return self.changes.has_updated()
        return False

    def updated_length(self):
        return self.changes.updated_length()

    def flush_deleted(self):
        return self.changes.flush_deleted()

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
    def column_changes(self):
        if self.changes:
            return self.changes.column_changes
        else:
            return None

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

    def create_slot(self, output_name, input_module, input_name):
        # No penalty in using a DataFrameSlot when a simple slot would have worked
        #if output_name==self._dataframe_slot:
        return DataFrameSlot(self, output_name, input_module, input_name)
        #return super(DataFrameModule, self).create_slot(output_name, input_module, input_name)

    def df(self):
        # from threading import current_thread
        # from traceback import print_stack
        # if current_thread().name != self.scheduler().thread_name:
        #     print('Current thread is: %s'% current_thread().ident)
        #     print_stack(limit=3)
            
        return self._df

    def get_data(self, name):
        if name==self._dataframe_slot:
            return self.df()
        return super(DataFrameModule, self).get_data(name)

    def get_progress(self):
        slot = self.get_input_slot(self._dataframe_slot)
        if slot is None or slot.data() is None:
            return (0,0)
        len = len(slot.data())
        if slot.has_created():
            return (len-slot.created_length(), len)
        # assume all has been processed
        return (len, len)



