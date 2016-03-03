from progressivis.core import Module, NIL
from progressivis.core.utils import indices_to_slice
from progressivis.core.index_diff import NIL_INDEX, index_diff, index_changes
import numpy as np
# TODO use the new RangeIndex when possible instead of explicit vector of indices

import logging
logger = logging.getLogger(__name__)

class ChangeManager(object):
    """Manage changes that occured in a DataFrame between runs.
    """
    def __init__(self,buffer_created=True,buffer_updated=False,buffer_deleted=False, manage_columns=True):
        self._buffer_created = buffer_created
        self._buffer_updated = buffer_updated
        self._buffer_deleted = buffer_deleted
        self._manage_columns = manage_columns
        self.reset()

    def reset(self):
        self.last_run = 0
        self.index = NIL_INDEX
        self.column_index = NIL_INDEX
        self._created = NIL
        self._updated = NIL
        self._deleted = NIL
        self._column_changes = None

    def next_state(self):
        if self._buffer_created and self.has_created():
            return Module.state_ready
        if self._buffer_updated and  self.has_updated():
            return Module.state_ready
        if self._buffer_deleted and  self.has_deleted():
            return Module.state_ready

        return Module.state_blocked

    def update(self, run_number, df):
        if df is None or run_number <= self.last_run:
            return
        index = df.index
        if index.has_duplicates:
            logger.error('cannot update changes, Index has duplicates')
        uc = df[Module.UPDATE_COLUMN]
        if self.last_run==0:
            self.index = index
            self.column_index = df.columns
            if self._buffer_created:
                self._created = self.index.values
            else:
                self._created = NIL
            self._updated = NIL
            self._deleted = NIL
            if self._manage_columns:
                self._column_changes = index_diff(created=df.columns, kept=NIL_INDEX, deleted=NIL_INDEX)
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
            if self._manage_columns:
                self._column_changes = index_changes(df.columns, self.column_index)
            self.column_index = df.columns
        self.last_run = run_number
        logger.debug('Updating for run_number %d: updated:%d/created:%d/deleted:%d',
                    run_number, len(self._updated), len(self._created), len(self._deleted))

    def manage_columns(self, v=True):
        self._manage_columns = v
        if not v:
            self._manage_columns = None

    @property
    def column_changes(self):
        return self._column_changes

    def buffer_created(self, v=True):
        self._buffer_created = v
        if not v:
            self._created = NIL

    def next_created(self, n=None):
        """
        Return at most n indices of newly created items. If n is not provided, return the
        indices of all the items that have been created.
        Note that if buffer_created is not set, only the items created in the last run
        will be returned
        """
        if not self._buffer_created:
            return NIL
        if n is None:
            n = len(self._created)
        ret, self._created = np.split(self._created, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return indices_to_slice(ret)

    def has_created(self):
        return len(self._created) != 0

    def created_length(self):
        return len(self._created)

    def buffer_updated(self, v=True):
        self._buffer_updated = v
        if not v:
            self._updated = NIL

    def next_updated(self, n=None):
        """
        Return at most n indices of newly updated items. If n is not provided, return the
        indices of all the items that have been updated.
        Note that if buffer_updated is not set, only the items updated in the last run
        will be returned
        """
        if not self._buffer_updated:
            return NIL
        if n is None:
            n = len(self._updated)
        ret, self._updated = np.split(self._updated, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return indices_to_slice(ret)

    def has_updated(self):
        return len(self._updated) != 0

    def updated_length(self):
        return len(self._updated)

    def buffer_deleted(self, v=True):
        self._buffer_deleted = v
        if not v:
            self._deleted = NIL

    def next_deleted(self, n=None):
        """
        Return at most n indices of newly deleted items. If n is not provided, return the
        indices of all the items that have been deleted.
        Note that if buffer_deleted is not set, only the items deleted in the last run
        will be returned
        """
        if not self._buffer_deleted:
            return NIL
        if n is None:
            n = len(self._deleted)
        ret, self._deleted = np.split(self._deleted, [n])
        # Try to return a slice instead of indices to avoid copying
        # arrays instead of creating views.
        return indices_to_slice(ret)

    def has_deleted(self):
        return len(self._deleted) != 0

    def deleted_length(self):
        return len(self._deleted)

