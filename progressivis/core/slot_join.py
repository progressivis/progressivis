from __future__ import annotations

from ..core.pintset import PIntSet
import operator
from functools import reduce
import weakref
from .slot import Slot

from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    from progressivis.core.module import Module


class SlotJoin:
    def __init__(self, module: Module, *slots: Slot):
        from ..table import BasePTable

        assert len(slots) > 0
        for slot in slots:
            assert isinstance(slot, Slot)
            assert isinstance(slot.data(), BasePTable)
        self._module_wr = weakref.ref(module)
        self._slots = slots

    def __enter__(self) -> SlotJoin:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None:
            self.manage_orphans()
        else:
            raise exc_type(exc_value)

    @property
    def _module(self) -> Module:
        module = self._module_wr()
        assert module is not None
        return module

    def next_created(self, step_size: int) -> PIntSet:
        changes_ = [slot.created.changes for slot in self._slots]
        common_ids = reduce(operator.and_, changes_)
        if not common_ids:
            return common_ids
        if len(common_ids) > step_size:
            common_ids = common_ids[:step_size]
        else:
            # always make a copy here (single slot issue)
            common_ids = common_ids[:]
        for slot in self._slots:
            slot.created.changes -= common_ids
        return common_ids

    def next_deleted(self, step_size: int, raw: bool = False) -> PIntSet:
        todo = step_size
        res = PIntSet()
        for slot in self._slots:
            indices = slot.deleted.next(length=todo, as_slice=False)
            res |= indices
            todo -= len(indices)
            if todo <= 0:
                break
        if not raw:
            existing = PIntSet(self._module.result.index)  # type: ignore
            return res & existing
        return res

    def next_updated(self, step_size: int, raw: bool = False) -> PIntSet:
        todo = step_size
        res = PIntSet()
        for slot in self._slots:
            indices = slot.updated.next(length=todo, as_slice=False)
            res |= indices
            todo -= len(indices)
            if todo <= 0:
                break
        if not raw:
            idx_ = [PIntSet(slot.data().index) for slot in self._slots]
            return reduce(operator.and_, idx_, res)
        return res

    def has_deleted(self, raw: bool = False) -> bool:
        changes_ = [slot.deleted.changes for slot in self._slots]
        res = reduce(operator.or_, changes_)
        if not raw:
            existing = PIntSet(self._module.result.index)  # type: ignore
            res &= existing
        return res != PIntSet()

    def has_updated(self, raw: bool = False) -> bool:
        changes_ = [slot.updated.changes for slot in self._slots]
        res = reduce(operator.or_, changes_)
        if not raw:
            idx_ = [PIntSet(slot.data().index) for slot in self._slots]
            res = reduce(operator.and_, idx_, res)
        return res != PIntSet()

    def has_created(self) -> bool:
        changes_ = [slot.created.changes for slot in self._slots]
        common_ids = reduce(operator.and_, changes_)
        return common_ids != PIntSet()

    def manage_orphans(self) -> None:
        for slot in self._slots:
            if slot.output_module.state > self._module.state_blocked:
                if not self.has_created():
                    for slot in self._slots:
                        slot.created.next(length=None, as_slice=False)
                        # TODO: add a callback for orphans processing
