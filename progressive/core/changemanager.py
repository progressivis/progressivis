from progressive.core.module import Module
import pandas as pd
import numpy as np

NIL = np.array([])

class ChangeManager(object):
    """Manage changes that accured in a DataFrame between runs.
    """
    def __init__(self, last_run=None):
        self.last_run = last_run
        self.index = pd.Index([])
        self.updated = NIL
        self.created = NIL
        self.deleted = NIL
        self.buffer = NIL

    def reset(self):
        self.last_run = None
        self.index = pd.Index([])
        self.updated = NIL
        self.created = NIL
        self.deleted = NIL
        self.buffer = NIL

    def update(self, run_number, df):
        if run_number <= self.last_run:
            return
        uc = df[Module.UPDATE_COLUMN]
        if self.last_run is None:
            self.index = df.index
            self.updated = self.index.values
            self.created = self.updated
            self.deleted = NIL
        else:
            self.updated = np.where(uc > self.last_run)[0]
            self.created = df.index.difference(self.index).values
            self.deleted = self.index.difference(df.index).values
            self.index = df.index
        self.last_run = run_number

    def buffer_updated(self):
        self.buffer = np.hstack([self.buffer, self.updated])

    def buffer_created(self):
        self.buffer = np.hstack([self.buffer, self.updated])

    def next_buffered(self, n):
        if len(self.buffer)==0:
            return NIL
        ret, self.buffer = np.split(self.buffer, [n])
        return ret

    def is_buffer_empty(self):
        return len(self.buffer)==0

    def next_state(self):
        if self.is_buffer_empty():
            return Module.state_blocked
        return Module.state_ready
