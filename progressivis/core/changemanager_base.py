"""
Base class for change managers.
Change managers are used by slots to maintain the list of changes in the data
structures they manage, typically a Table, a Column, or a bitmap.
"""
from __future__ import annotations

import logging
from .index_update import IndexUpdate
from .bitmap import bitmap
import weakref as wr

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Any, Optional, Union, TYPE_CHECKING, overload

if TYPE_CHECKING:
    from .slot import Slot


logger = logging.getLogger(__name__)


class BaseChangeManager:
    "Base class for change managers"

    def __init__(
        self,
        slot: Slot,
        buffer_created: bool = True,
        buffer_updated: bool = False,
        buffer_deleted: bool = False,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ) -> None:
        _ = slot
        self._row_changes = IndexUpdate()
        self._selection_changes = IndexUpdate()
        self._base: Optional[_accessor] = None
        self._selection: Optional[_accessor] = None
        # The bitmaps are shared between _row_changes and the buffers.
        # To remain shared, they should never be assigned to, only updated.
        self._created = ChangeBuffer(buffer_created, self._row_changes.created)
        self._updated = ChangeBuffer(buffer_updated, self._row_changes.updated)
        self._deleted = ChangeBuffer(buffer_deleted, self._row_changes.deleted)
        self._exposed = ChangeBuffer(buffer_exposed, self._selection_changes.created)
        self._masked = ChangeBuffer(buffer_masked, self._selection_changes.deleted)
        self._last_update: int = 0

    @property
    def column_changes(self) -> Optional[Any]:
        "Return information for columns changed"
        return None

    @property
    def created(self) -> ChangeBuffer:
        "Return information of items created"
        return self._created

    @property
    def updated(self) -> ChangeBuffer:
        "Return information of items updated"
        return self._updated

    @property
    def deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        return self._deleted

    @property
    def base(self) -> _accessor:
        if self._base is None:
            self._base = _base_accessor(self)
        return self._base

    @property
    def selection(self) -> _accessor:
        if self._selection is None:
            self._selection = _selection_accessor(self)
        return self._selection

    def reset(self, mid: str) -> None:
        """
        Reset the change manager so changes will come as if the managed data
        was freshly created.
        """
        # pylint: disable=unused-argument
        self._last_update = 0
        self.clear()
        logger.debug("reset(%d)", self._last_update)

    @property
    def row_changes(self) -> IndexUpdate:
        "Return the IndexUpdate keeping track of the row changes"
        return self._row_changes

    @property
    def mask_changes(self) -> IndexUpdate:
        "Return the IndexUpdate keeping track of the mask changes"
        return self._selection_changes

    def has_buffered(self) -> bool:
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return (
            self.created.any()
            or self.updated.any()
            or self.deleted.any()
            or self.base.deleted.any()
            or self.selection.deleted.any()
        )

    def last_update(self) -> int:
        "Return the date of the last update"
        return self._last_update

    def update(self, run_number: int, data: Any, mid: str) -> None:  # pragma no cover
        """
        Compute the changes from the last_update to the specified run_number.
        """
        pass

    def clear(self) -> None:
        """
        removed all the buffered information from the
        created/updated/deleted/exposed/masked information
        """
        self._row_changes.clear()
        self._selection_changes.clear()

    def dump(self) -> None:
        import inspect

        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        print("caller name:", calframe[1][3])
        print("created", self._created.changes)
        print("updated", self._updated.changes)
        print("deleted", self._deleted.changes)
        print("exposed", self._exposed.changes)
        print("masked", self._masked.changes)


def _next(bm: bitmap, length: Optional[int], as_slice: bool) -> Union[bitmap, slice]:
    if length is None:
        length = len(bm)
    ret = bm.pop(length)
    if as_slice:
        return ret.to_slice_maybe()
    return ret


class ChangeBuffer:
    def __init__(self, default: bool, changes: bitmap):
        self.buffer = default
        self.changes = changes

    def set_buffered(self, v: bool = True) -> None:
        "Set if data is buffered"
        self.buffer = v
        if not v:
            self.changes.clear()

    def clear(self) -> None:
        "Clear the buffer"
        self.changes.clear()

    def __len__(self) -> int:
        "Return the buffer length"
        return len(self.changes)

    def length(self) -> int:
        "Return the buffer length"
        return len(self.changes)

    def any(self) -> bool:
        "Return True if there is anything in the buffer"
        return self.buffer and len(self.changes) != 0

    @overload
    def next(
        self, length: Optional[int] = None, *, as_slice: Literal[True] = True
    ) -> slice:
        ...

    @overload
    def next(self, length: Optional[int] = None, *, as_slice: Literal[False]) -> bitmap:
        ...

    @overload
    def next(self, length: Optional[int], *, as_slice: bool) -> Union[bitmap, slice]:
        ...

    def next(self, length: Optional[int] = None, *, as_slice: bool = True) -> Any:
        "Return the next items in the buffer"
        if not self.buffer:
            return None
        return _next(self.changes, length, as_slice)

    def pop(self, bm: bitmap) -> None:
        "Remove a set of items from the buffer"
        if self.buffer:
            self.changes -= bm

    def push(self, bm: bitmap) -> None:
        "Adds a sset of items in the buffer"
        if self.buffer:
            self.changes |= bm


EMPTY_BUFFER = ChangeBuffer(False, bitmap())


class _accessor:
    def __init__(self, parent: BaseChangeManager):
        self._parent_wr: wr.ReferenceType[BaseChangeManager] = wr.ref(parent)

    @property
    def _parent(self) -> Union[None, BaseChangeManager]:
        return self._parent_wr()

    @property
    def created(self) -> ChangeBuffer:
        "Return information of items created"
        assert self._parent
        return self._parent._created

    @property
    def updated(self) -> ChangeBuffer:
        "Return information of items updated"
        assert self._parent
        return self._parent._updated

    @property
    def deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        assert self._parent
        return self._parent._deleted


class _base_accessor(_accessor):
    @property
    def created(self) -> ChangeBuffer:
        "Return information of items created"
        assert self._parent
        return self._parent._created

    @property
    def updated(self) -> ChangeBuffer:
        "Return information of items updated"
        assert self._parent
        return self._parent._updated

    @property
    def deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        assert self._parent
        return self._parent._deleted


class _selection_accessor(_accessor):
    @property
    def created(self) -> ChangeBuffer:
        "Return information of items created"
        assert self._parent
        return self._parent._exposed

    @property
    def deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        assert self._parent
        return self._parent._masked
