"""
Base class for change managers.
Change managers are used by slots to maintain the list of changes in the data
structures they manage, typically a Table, a Column, or a bitmap.
"""

import logging
from .index_update import IndexUpdate
from .bitmap import bitmap

logger = logging.getLogger(__name__)


class BaseChangeManager(object):
    "Base class for change managers"

    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=False,
                 buffer_deleted=False,
                 buffer_exposed=False,
                 buffer_masked=False):
        _ = slot
        self._row_changes = IndexUpdate()
        self._mask_changes = IndexUpdate()
        # The bitmaps are shared between _row_changes and the buffers.
        # To remain shared, they should never be assigned to, only updated.
        self._created = _buffer(buffer_created, self._row_changes.created)
        self._updated = _buffer(buffer_updated, self._row_changes.updated)
        self._deleted = _buffer(buffer_deleted, self._row_changes.deleted)
        self._exposed = _buffer(buffer_exposed, self._mask_changes.created)
        self._masked = _buffer(buffer_masked, self._mask_changes.deleted)
        self._last_update = 0

    @property
    def column_changes(self):
        "Return information for columns changed"
        return None

    @property
    def created(self):
        "Return information of items created"
        return self._created

    @property
    def updated(self):
        "Return information of items updated"
        return self._updated

    @property
    def deleted(self):
        "Return information of items deleted"
        return self._deleted

    def reset(self, name=None):
        """
        Reset the change manager so changes will come as if the managed data
        was freshly created.
        """
        # pylint: disable=unused-argument
        self._last_update = 0
        self.clear()
        logger.debug('reset(%d)', self._last_update)

    @property
    def exposed(self):
        "Return information of items exposed"
        return self._exposed

    @property
    def masked(self):
        "Return information of items masked"
        return self._masked

    @property
    def row_changes(self):
        "Return the IndexUpdate keeping track of the row changes"
        return self._row_changes

    @property
    def mask_changes(self):
        "Return the IndexUpdate keeping track of the mask changes"
        return self._mask_changes

    def has_buffered(self):
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return (self.created.any() or self.updated.any() or self.deleted.any()
                or self.exposed.any() or self.masked.any())

    def last_update(self):
        "Return the date of the last update"
        return self._last_update

    def update(self, run_number, data, mid):  # pragma no cover
        """
        Compute the changes from the last_update to the specified run_number.
        """
        pass

    def clear(self):
        """
        removed all the buffered information from the
        created/updated/deleted/exposed/masked information
        """
        self._row_changes.clear()
        self._mask_changes.clear()


def _next(bm, length, as_slice):
    if length is None:
        length = len(bm)
    ret = bm.pop(length)
    if as_slice:
        ret = ret.to_slice_maybe()
    return ret


class _buffer(object):
    def __init__(self, default, changes):
        self.buffer = default
        self.changes = changes

    def set_buffered(self, v=True):
        "Set if data is buffered"
        self.buffer = v
        if not v:
            self.changes.clear()

    def clear(self):
        "Clear the buffer"
        self.changes.clear()

    def __len__(self):
        "Return the buffer length"
        return len(self.changes)

    def length(self):
        "Return the buffer length"
        return len(self.changes)

    def any(self):
        "Return True if there is anything in the buffer"
        return self.buffer and len(self.changes) != 0

    def next(self, length=None, as_slice=True):
        "Return the next items in the buffer"
        if not self.buffer:
            return None
        return _next(self.changes, length, as_slice)

    def pop(self, bm):
        "Remove a set of items from the buffer"
        if self.buffer:
            self.changes -= bm

    def push(self, bm):
        "Adds a sset of items in the buffer"
        if self.buffer:
            self.changes |= bm


EMPTY_BUFFER = _buffer(False, bitmap())
