from . import Module, NIL, ProgressiveError
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger(__name__)

def maybe_slice(array):
    if len(array)==0:
        return slice(0,0)
    s = e = None
    array.sort()
    for i in array:
        if s is None:
            s = e = i
        elif i==e or i==e+1:
            e=i
        else:
            return array # not sliceable
    return slice(s, e+1)

CM_CREATED = 1
CM_UPDATED = 2
CM_DELETED = 4

class ChangeManagerBase(object):
    
    def __init__(self):
        self.reset()
        self._buffer_created = True
        self._buffer_updated = False
        self._buffer_deleted = False

    def reset(self):
        logger.info('Reseting history')
        self.last_run = 0
        self.index = pd.Index([])
        self._created = None
        self._updated = None

    def update(self, run_number, df):
        raise ProgressiveError('Update not implemented')

    def next_state(self, buffer=CM_CREATED):
        if buffer&CM_CREATED and self.has_created():
            return Module.state_ready
        if buffer&CM_UPDATED and self.has_updated():
            return Module.state_ready
        if buffer&CM_DELETED and self.has_deleted():
            return Module.state_ready

        return Module.state_blocked

    def buffer_created(self, v=True):
        self._buffer_created = v

    def next_created(self, n=None):
        raise ProgressiveError('Not implemented')

    def has_created(self):
        raise ProgressiveError('Not implemented')

    def created_length(self):
        raise ProgressiveError('Not implemented')
    
    def buffer_updated(self, v=True):
        self._buffer_updated = v

    def next_updated(self, n=None):
        raise ProgressiveError('Not implemented')

    def has_updated(self):
        raise ProgressiveError('Not implemented')

    def updated_length(self):
        raise ProgressiveError('Not implemented')

    def buffer_deleted(self, v=True):
        self._buffer_deleted = v

    def next_deleted(self, n=None):
        raise ProgressiveError('Not implemented')
    
    def has_deleted(self):
        raise ProgressiveError('Not implemented')

    def deleted_length(self):
        raise ProgressiveError('Not implemented')

# class SimpleChangeManager(ChangeManagerBase):
#     def __init__(self):
#         super(SimpleChangeManager,self).__init__()
    
#     def reset(self):
#         super(SimpleChangeManager,self).reset()
#         self._buffer = slice(0, 0)

#     def update(self, run_number, df):
#         if run_number <= self.last_run:
#             return True
#         uc = df[Module.UPDATE_COLUMN]
#         self._buffered = False

#         nlen = len(df)
#         if self.last_run==0:
#             self._created = slice(0, nlen)
#         else:
#             olen = len(self.index)
#             end = slice(olen,olen+nlen)
#             if not (uc[end] > self.last_run).all():
#                 return False # something else happened
#             self._created = slice(self._created.start, end.stop)
#         self.index = df.index
#         self.last_run = run_number
#         logger.info('Updating for run_number %d: created:%s',
#                     run_number, self._created)
#         self.buffer()
#         return True

#     def buffer(self):
#         if self._buffered:
#             return
#         self._buffered = True
#         if self._created is None or self._created.start==self._created.stop:
#             return
#         self._buffer = slice(self._buffer.start, self._created.stop)

#     def next_buffered(self, n, as_ranges=False):
#         if (self._buffer.start+n) >= self._buffer.stop:
#            ret = self._buffer
#         else:
#             ret = slice(self._buffer.start, self._buffer.start+n)
#         self._buffer = slice(ret.stop, self._buffer.stop)
#         return ret

#     def is_buffer_empty(self):
#         return self._buffer.start >= self._buffer.stop
    

class ChangeManager(ChangeManagerBase):
    """Manage changes that occured in a DataFrame between runs.
    """
    def __init__(self,buffer_created=True,buffer_updated=False,buffer_deleted=False):
        super(ChangeManager,self).__init__() # calls reset()
        self._buffer_created = buffer_created
        self._buffer_updated = buffer_updated
        self._buffer_deleted = buffer_deleted

    def reset(self):
        super(ChangeManager,self).reset()
        self._created = NIL
        self._updated = NIL
        self._deleted = NIL
        self._buffer = NIL

    def update(self, run_number, df):
        if df is None or run_number <= self.last_run:
            return
        index = df.index
        if index.has_duplicates:
            logger.error('Index has duplicates')
            import pdb
            pdb.set_trace()
        uc = df[Module.UPDATE_COLUMN]
        if self.last_run==0:
            self.index = index
            self._created = self.index.values
            self._updated = NIL
            self._deleted = NIL
        else:
            # Simle case 1: nothing deleted
            l1 = len(self.index)
            l2 = len(index) 
            if l1 <= l2 and np.array_equal(self.index,index[0:l1]):
                deleted = NIL
                updated = np.where(uc[0:l1] > self.last_run)[0]
                created = index[l1:]
            #TODO: These computations are potentially expensive
            # later, optimize them by testing simple cases first
            # such as only created items, or only updated items
            else:
                deleted = self.index.difference(index).values
                updated = np.where(uc[self.index] > self.last_run)[0]
                created = index.difference(self.index).values

            if self._buffer_created:
                # For created items still buffered, we can ignore that they've been updated
                updated = np.setdiff1d(updated, self._created)
                self._created = np.union1d(np.setdiff1d(self._created, deleted), created)
            else:
                self._created = created
            
            if self._buffer_deleted:
                self._deleted = np.union1d(self._deleted, deleted)
            else:
                self._deleted = deleted

            if self._buffer_updated:
                self._updated = np.union1d(np.setdiff1d(self._updated, deleted), updated)
            else:
                self._updated = updated

            self.index = index
        self.last_run = run_number
        logger.info('Updating for run_number %d: updated:%d/created:%d/deleted:%d',
                    run_number, len(self._updated), len(self._created), len(self._deleted))

    def next_created(self, n=None):
        """
        Return at most n indices of newly created items. If n is not provided, return the
        indices of all the items that have been created.
        Note that if buffer_created is not set, only the items created in the last run
        will be returned
        """
        if n is None:
            n = len(self._created)
        ret, self._created = np.split(self._created, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return maybe_slice(ret)

    def has_created(self):
        return len(self._created) != 0

    def created_length(self):
        return len(self._created)

    def next_updated(self, n=None):
        """
        Return at most n indices of newly updated items. If n is not provided, return the
        indices of all the items that have been updated.
        Note that if buffer_updated is not set, only the items updated in the last run
        will be returned
        """
        if n is None:
            n = len(self._updated)
        ret, self._updated = np.split(self._updated, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return maybe_slice(ret)

    def has_updated(self):
        return len(self._updated) != 0

    def updated_length(self):
        return len(self._updated)

    def next_deleted(self, n=None):
        """
        Return at most n indices of newly deleted items. If n is not provided, return the
        indices of all the items that have been deleted.
        Note that if buffer_deleted is not set, only the items deleted in the last run
        will be returned
        """
        if n is None:
            n = len(self._deleted)
        ret, self._deleted = np.split(self._deleted, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return maybe_slice(ret)

    def has_deleted(self):
        return len(self._deleted) != 0

    def deleted_length(self):
        return len(self._deleted)
