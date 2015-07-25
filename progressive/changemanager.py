from progressive.module import Module
import pandas as pd
import numpy as np

NIL = np.array([])

class ChangeManager(object):
    """Manage changes that accured in a DataFrame between runs.
    """
    def __init__(self, last_time=None):
        self.last_time = last_time
        self.index = pd.Index([])
        self.updated = NIL
        self.created = NIL
        self.deleted = NIL
        self.buffer = NIL

    def update(self, time, df):
        if time <= self.last_time:
            return
        uc = df[Module.UPDATE_COLUMN]
        if self.last_time is None:
            self.index = df.index
            self.updated = self.index.values
            self.created = self.updated
            self.deleted = NIL
        else:
            self.updated = np.where(uc > self.last_time)[0]
            self.created = df.index.difference(self.index).values
            self.deleted = self.index.difference(df.index).values
            self.index = df.index
        self.last_time = time

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
