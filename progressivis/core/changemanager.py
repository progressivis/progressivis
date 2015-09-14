from progressivis.core.module import Module
from progressivis.core.common import NIL
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def ranges(array):
    """Convert an array of indices into a list of ranges (pairs).

>>> lst = [1, 5, 6, 7, 12, 15, 16, 17, 18, 30]
>>> print repr(ranges(lst))
[(1, 1), (5, 7), (12, 12), (15, 18), (30, 30)]
    """
    s = e = None
    r = []
    array.sort()
    for i in array:
        if s is None:
            s = e = i
        elif i == e or i == e + 1:
            e = i
        else:
            r.append((s, e))
            s = e = i
    if s is not None:
        r.append((s, e))
    return r

class ChangeManagerBase(object):
    def __init__(self):
        self.reset()

    def reset(self):
        logger.info('Reseting history')
        self.last_run = None
        self.index = pd.Index([])
        self._buffer = None
        self._buffered = False
        self._created = None

    def update(self, run_number, df):
        raise ProgressiveError('Update not implemented')

    def buffer(self):
        raise ProgressiveError('buffer not implemented')

    def next_buffered(self, n, as_ranges=True):
        raise ProgressiveError('next_buffered not implemented')
        
    def is_buffer_empty(self):
        raise ProgressiveError('is_buffer_empty not implemented')

    def next_state(self):
        raise ProgressiveError('next_state not implemented')

    def next_state(self):
        if self.is_buffer_empty():
            return Module.state_blocked
        return Module.state_ready

class SimpleChangeManager(ChangeManagerBase):
    def __init__(self):
        super(SimpleChangeManager,self).__init__()
    
    def reset(self):
        super(SimpleChangeManager,self).reset()
        self._buffer = slice(0, 0)

    def update(self, run_number, df):
        if self.last_run is not None and run_number <= self.last_run:
            return True
        uc = df[Module.UPDATE_COLUMN]
        self._buffered = False

        nlen = len(df)
        if self.last_run is None:
            self._created = slice(0, nlen)
        else:
            olen = len(self.index)
            end = slice(olen,olen+nlen)
            if not (uc[end] > self.last_run).all():
                return False # something else happened
            self._created = slice(self._created.start, end.stop)
        self.index = df.index
        self.last_run = run_number
        logger.info('Updating for run_number %d: created:%s',
                    run_number, self._created)
        self.buffer()
        return True

    def buffer(self):
        if self._buffered:
            return
        self._buffered = True
        if self._created is None or self._created.start==self._created.stop:
            return
        self._buffer = slice(self._buffer.start, self._created.stop)

    def next_buffered(self, n, as_ranges=False):
        if (self._buffer.start+n) >= self._buffer.stop:
           ret = self._buffer
        else:
            ret = slice(self._buffer.start, self._buffer.start+n)
        self._buffer = slice(ret.stop, self._buffer.stop)
        return ret

    def is_buffer_empty(self):
        return self._buffer.start >= self._buffer.stop
    

class ChangeManager(ChangeManagerBase):
    """Manage changes that accured in a DataFrame between runs.
    """
    def __init__(self):
        super(ChangeManager,self).__init__() # calls reset()

    def reset(self):
        super(ChangeManager,self).reset()
        self._updated = NIL
        self._created = NIL
        self._deleted = NIL
        self._buffer = NIL

    def update(self, run_number, df):
        if self.last_run is not None and run_number <= self.last_run:
            return True
        uc = df[Module.UPDATE_COLUMN]
        self._buffered = False
        #TODO flush buffer containing data invalidated since the last run.
        if self.last_run is None:
            self.index = df.index
            self._updated = self.index.values
            self._created = self._updated
            self._deleted = NIL
        else:
            self._updated = np.where(uc > self.last_run)[0]
            self._created = df.index.difference(self.index).values
            self._deleted = self.index.difference(df.index).values
            self.index = df.index
        self.last_run = run_number
        logger.info('Updating for run_number %d: updated:%d/created:%d/deleted:%d',
                    run_number, len(self._updated), len(self._created), len(self._deleted))
        return True

    def buffer_updated(self):
        if self._buffered:
            return
        logger.info('Filling-up buffer for updated')
        self._buffered = True
        if len(self._updated)!=0:
            self._buffer = np.hstack([self._buffer, self._updated])

    def buffer(self):
        if self._buffered:
            return
        logger.info('Filling-up buffer for created')
        self._buffered = True
        if len(self._created) != 0:
            self._buffer = np.hstack([self._buffer, self._created])

    def next_buffered(self, n, as_ranges=False):
        if len(self._buffer)==0:
            logger.info('Returning null buffer')
            return NIL
        if n >= len(self._buffer):
            ret = self._buffer
            self._buffer = NIL
        else:
            ret, self._buffer = np.split(self._buffer, [n])
        logger.info('Returning buffer of %d/%d', len(ret), len(self._buffer))
        if as_ranges:
            return ranges(ret)
        return ret

    def is_buffer_empty(self):
        return len(self._buffer)==0

