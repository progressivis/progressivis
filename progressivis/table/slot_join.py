from ..core.bitmap import bitmap
from .. import Slot
from ..table import BaseTable
import operator
from functools import reduce
import weakref


class SlotJoin:
    def __init__(self, module, *slots):
        assert len(slots) > 0
        for slot in slots:
            assert isinstance(slot, Slot)
            assert isinstance(slot.data(), BaseTable)
        self._module_wr = weakref.ref(module)
        self._slots = slots

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.manage_orphans()
        else:
            raise exc_type(exc_value)

    @property
    def _module(self):
        return self._module_wr()

    def next_created(self, step_size):
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

    def next_deleted(self, step_size, raw=False):
        todo = step_size
        res = bitmap()
        for slot in self._slots:
            indices = slot.deleted.next(todo, as_slice=False)
            res |= indices
            todo -= len(indices)
            if todo <= 0:
                break
        if not raw:
            existing = bitmap(self._module.result.index)
            return res & existing
        return res

    def next_updated(self, step_size, raw=False):
        todo = step_size
        res = bitmap()
        for slot in self._slots:
            indices = slot.updated.next(todo, as_slice=False)
            res |= indices
            todo -= len(indices)
            if todo <= 0:
                break
        if not raw:
            idx_ = [bitmap(slot.data().index) for slot in self._slots]
            return reduce(operator.and_, idx_, res)
        return res

    def has_deleted(self, raw=False):
        changes_ = [slot.deleted.changes for slot in self._slots]
        res = reduce(operator.or_, changes_)
        if not raw:
            existing = bitmap(self._module.result.index)
            res &= existing
        return res != bitmap()

    def has_updated(self, raw=False):
        changes_ = [slot.updated.changes for slot in self._slots]
        res = reduce(operator.or_, changes_)
        if not raw:
            idx_ =  [bitmap(slot.data().index) for slot in self._slots]
            res = reduce(operator.and_, idx_, res)
        return res != bitmap()

    def has_created(self):
        changes_ = [slot.created.changes for slot in self._slots]
        common_ids = reduce(operator.and_, changes_)
        return common_ids != bitmap()

    def manage_orphans(self):
        for slot in self._slots:
            if slot.output_module.state > self._module.state_blocked:
                if not self.has_created():
                    for slot in self._slots:
                        slot.created.next() # TODO: add a callback for orphans processing
