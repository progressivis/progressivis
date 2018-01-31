from __future__ import absolute_import, division, print_function

from abc import ABCMeta, abstractmethod
import logging
from .index_update import IndexUpdate
from .bitmap import bitmap

logger = logging.getLogger(__name__)


class BaseChangeManager(object):
    __metaclass__ = ABCMeta

    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False,
                 manage_columns=True):
        # pylint: disable=unused-argument
        self._row_changes = IndexUpdate()
        # The bitmaps are shared between _row_changes and the buffers.
        # To remain shared, they should never be assigned to, only updated.
        self._created = buffer(buffer_created, self._row_changes.created)
        self._updated = buffer(buffer_updated, self._row_changes.updated)
        self._deleted = buffer(buffer_deleted, self._row_changes.deleted)
        self._manage_columns = manage_columns
        self._last_update = 0

    @property
    def created(self):
        return self._created

    @property
    def updated(self):
        return self._updated

    @property
    def deleted(self):
        return self._deleted

    def reset(self, mid=None):
        # pylint: disable=unused-argument
        self._last_update = 0
        self.clear()
        logger.debug('reset(%d)', self._last_update)

    @property
    def row_changes(self):
        return self._row_changes

    def has_buffered(self):
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return self.created.any() or self.updated.any() or self.deleted.any()

    def last_update(self):
        return self._last_update

    @abstractmethod
    def update(self, run_number, data, mid=None):
        """
        Compute the changes from the last_update to the specified run_number.
        """
        pass

    def clear(self):
        self._row_changes.clear()


def _next(bm, length, as_slice):
    if length is None:
        length = len(bm)
    ret = bm.pop(length)
    if as_slice:
        ret = ret.to_slice_maybe()
    return ret


class buffer(object):
    def __init__(self, default, changes):
        self.buffer = default
        self.changes = changes

    def set_buffered(self, v=True):
        self.buffer = v
        if not v:
            self.changes.clear()

    def clear(self):
        self.changes.clear()

    def __len__(self):
        return len(self.changes)

    def length(self):
        return len(self.changes)

    def any(self):
        return self.buffer and len(self.changes) != 0

    def next(self, length=None, as_slice=True):
        if not self.buffer:
            return None
        return _next(self.changes, length, as_slice)

    def pop(self, bm):
        if self.buffer:
            self.changes -= bm

    def push(self, bm):
        if self.buffer:
            self.changes |= bm

EMPTY_BUFFER = buffer(False, bitmap())
