from progressivis.core.module import Module
from progressivis.core.common import NIL
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

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
        self.buffered = False

    def reset(self):
        self.last_run = None
        self.index = pd.Index([])
        self.updated = NIL
        self.created = NIL
        self.deleted = NIL
        self.buffer = NIL
        self.buffered = False

    def update(self, run_number, df):
        if run_number <= self.last_run:
            return
        uc = df[Module.UPDATE_COLUMN]
        self.buffered = False
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
        logger.info('Updating for run_number %d: updated:%d/created:%d/deleted:%d',
                    run_number, len(self.updated), len(self.created), len(self.deleted))

    def buffer_updated(self):
        if not self.buffered:
            logger.info('Filling-up buffer for updated')
            self.buffered = True
            if len(self.updated)!=0:
                self.buffer = np.hstack([self.buffer, self.updated])

    def buffer_created(self):
        if not self.buffered:
            logger.info('Filling-up buffer for created')
            self.buffered = True
            if len(self.created) != 0:
                self.buffer = np.hstack([self.buffer, self.created])

    def next_buffered(self, n):
        if len(self.buffer)==0:
            logger.info('Returning null buffer')
            return NIL
        if n >= len(self.buffer):
            ret = self.buffer
            self.buffer = NIL
        else:
            ret, self.buffer = np.split(self.buffer, [n])
        logger.info('Returning buffer of %d/%d', len(ret), len(self.buffer))
        return ret

    def is_buffer_empty(self):
        return len(self.buffer)==0

    def next_state(self):
        if self.is_buffer_empty():
            return Module.state_blocked
        return Module.state_ready
