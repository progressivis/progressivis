"Change manager for SelectedTable"
from __future__ import annotations

from .table_base import TableSelectedView
from ..core.changemanager_base import (
    BaseChangeManager,
    _accessor,
    _base_accessor,
    ChangeBuffer,
)
from ..core.changemanager_bitmap import BitmapChangeManager
from ..core.slot import Slot
from ..core.bitmap import bitmap

import sys

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from typing import Any, Optional, Union, overload


class FakeSlot(Slot):
    # pylint: disable=too-few-public-methods
    "Fake slot to provide data to inner change manager"
    #    __fields__ = "data"

    def __init__(self, data: Any) -> None:
        self._data = data

    def data(self) -> Any:
        return self._data


class _double_buffer(ChangeBuffer):
    def __init__(self, first: ChangeBuffer, second: ChangeBuffer):
        self._first = first
        self._second = second
        self.buffer: bool = True

    def clear(self) -> None:
        "Clear the buffer"
        self._first.clear()
        self._second.clear()

    @overload
    def make_result(self, res: bitmap, as_slice: Literal[True]) -> slice:
        ...

    @overload
    def make_result(self, res: bitmap, as_slice: Literal[False]) -> bitmap:
        ...

    @overload
    def make_result(self, res: bitmap, as_slice: bool) -> Union[bitmap, slice]:
        ...

    def make_result(self, res: bitmap, as_slice: bool) -> Any:
        return res.to_slice_maybe() if as_slice else res

    def __len__(self) -> int:
        "Return the buffer length"
        return len(self._first.changes | self._second.changes)

    def length(self) -> int:
        "Return the buffer length"
        return self.__len__()

    def any(self) -> bool:  # TODO: improve
        "Return True if there is anything in the buffer"
        return self._first.any() or self._second.any()

    def next(self, length: Optional[int] = None, *, as_slice: bool = True) -> Any:
        if length is None:
            length = len(self._first.changes | self._second.changes)
        "Return the next items in the buffer"
        acc = bitmap()
        while length and self._first.any():
            # prevents to return second ids twice
            new_ids = (
                self._first.next(length=length, as_slice=False) - self._second.changes
            )
            length -= len(new_ids)
            acc |= new_ids
        if length and self._second.any():
            acc |= self._second.next(length=length, as_slice=False)
        return self.make_result(acc, as_slice)

    @property
    def all_changes(self) -> bitmap:
        return self._first.changes | self._second.changes

    def remove_from_all(self, ids: bitmap) -> None:
        self._first.changes -= ids
        self._second.changes -= ids


class TableSelectedChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a TableSelectedView between runs.
    """

    def __init__(
        self,
        slot: Slot,
        buffer_created: bool = True,
        buffer_updated: bool = False,
        buffer_deleted: bool = False,
        buffer_exposed: bool = False,
        buffer_masked: bool = False,
    ) -> None:
        data = slot.data()
        assert isinstance(data, TableSelectedView)
        bmslot = FakeSlot(data.index)  # not required formally
        tbslot = FakeSlot(data._base)
        super().__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted,
            buffer_exposed,
            buffer_masked,
        )
        self._mask_cm = BitmapChangeManager(
            bmslot,
            buffer_created=buffer_exposed,
            buffer_updated=False,
            buffer_deleted=buffer_masked,
            buffer_exposed=False,
            buffer_masked=False,
        )
        from .changemanager_table import TableChangeManager

        self._table_cm = TableChangeManager(
            tbslot,
            buffer_created=buffer_created,
            buffer_updated=buffer_updated,
            buffer_deleted=buffer_deleted,
            buffer_exposed=False,
            buffer_masked=False,
        )
        self._selection_changes = self._mask_cm._row_changes

    def reset(self, mid: str) -> None:
        super().reset(mid)
        self._mask_cm.reset(mid)
        self._table_cm.reset(mid)

    def update(self, run_number: int, data: Any, mid: str) -> None:
        assert isinstance(data, TableSelectedView)
        if data is None or (run_number != 0 and run_number <= self._last_update):
            return
        table = data.base
        selection = data.index
        self._mask_cm.update(run_number, selection, mid)
        self._table_cm.update(run_number, table, mid)
        table_changes = self._table_cm.row_changes
        # Mask table changes with current selection.
        table_changes.created &= selection
        table_changes.updated &= selection
        table_changes.deleted &= selection | self._selection_changes.deleted

        self._row_changes.combine(
            table_changes,
            update_created=self.created.buffer,
            update_updated=self.updated.buffer,
            update_deleted=self.deleted.buffer,
        )
        table_changes.clear()
        self._last_update = run_number

    @property
    def created(self) -> ChangeBuffer:
        "Return information of items created"
        return _double_buffer(first=self._created, second=self._mask_cm._created)

    @property
    def updated(self) -> ChangeBuffer:
        "Return information of items updated"
        return self._updated

    @property
    def deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        return _double_buffer(first=self._deleted, second=self._mask_cm._deleted)

    @property
    def selection(self) -> _accessor:
        if self._selection is None:
            self._selection = _base_accessor(self._mask_cm)
        return self._selection

    @property
    def perm_deleted(self) -> ChangeBuffer:
        "Return information of items deleted"
        return self._deleted

    """
    @property
    def row_changes(self):
        "Return the IndexUpdate keeping track of the row changes"
        return self._table_cm._row_changes

    @property
    def mask_changes(self):
        "Return the IndexUpdate keeping track of the mask changes"
        return self._mask_cm._row_changes
    """

    def has_buffered(self) -> bool:
        """
        If the change manager has something buffered, then the module is
        ready to run immediately.
        """
        return self._table_cm.has_buffered() or self._mask_cm.has_buffered()

    def last_update(self) -> int:
        "Return the date of the last update"
        return max(self._table_cm._last_update, self._mask_cm._last_update)


Slot.add_changemanager_type(TableSelectedView, TableSelectedChangeManager)
