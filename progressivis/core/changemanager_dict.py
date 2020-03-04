
#from .bitmap import bitmap
from .index_update import IndexUpdate
from .changemanager_base import BaseChangeManager
from .slot import Slot
import copy

class DictChangeManager(BaseChangeManager):
    """
    Manage changes that occured in a DataFrame between runs.
    """
    def __init__(self,
                 slot,
                 buffer_created=True,
                 buffer_updated=True,
                 buffer_deleted=True):
        super(DictChangeManager, self).__init__(
            slot,
            buffer_created,
            buffer_updated,
            buffer_deleted)
        self._last_dict = None

    def reset(self, name=None):
        super(DictChangeManager, self).reset(name)
        self._last_bm = None

    def compute_updates(self, data):
        last_dict = self._last_dict
        changes = IndexUpdate()
        if last_dict is None:
            if self.created.buffer:
                changes.created.update(data.ids)
        else:
            if self.created.buffer:
                changes.created.update(data.new_indices(last_dict))
            if self.updated.buffer:
                changes.updated.update(data.updated_indices(last_dict))
            if self.deleted.buffer:
                changes.deleted.update(data.deleted_indices(last_dict))
        self._last_dict = copy.copy(data)
        return changes

    def update(self, run_number, data, mid):
        # pylint: disable=unused-argument
        assert isinstance(data, dict)
        if data is None or (run_number != 0 and
                            run_number <= self._last_update):
            return

        changes = self.compute_updates(data)
        self._last_update = run_number
        self._row_changes.combine(changes,
                                  self.created.buffer,
                                  self.updated.buffer,
                                  self.deleted.buffer)


Slot.add_changemanager_type(dict, DictChangeManager)
