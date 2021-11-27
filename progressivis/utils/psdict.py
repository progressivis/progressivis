from ..core.bitmap import bitmap
from ..core.index_update import IndexUpdate
from ..table.dshape import dshape_extract
import numpy as np


class PsDict(dict):
    "progressive dictionary"
    def __init__(self, other=None, **kwargs):
        if other is not None:
            # useful when keys are not varname-alike
            # one can use both (other and kwargs)
            # if keys are not duplicate
            assert isinstance(other, dict)
        else:
            other = {}
        super().__init__(other, **kwargs)
        self._index = None
        self._deleted = {}
        self._inverse = None
        self._inverse_del = None
        self.changes = None

    def compute_updates(self, start, now, mid=None, cleanup=True):
        assert False, "compute_updates should not be called on PsDict"
        if self.changes:
            updates = self.changes.compute_updates(start, now, mid,
                                                   cleanup=cleanup)
            if updates is None:
                updates = IndexUpdate(created=bitmap(self.ids))
            return updates
        return None

    def fill(self, val):
        for k in self.keys():
            self[k] = val

    @property
    def array(self):
        return np.array(list(self.values()))

    @property
    def as_row(self):
        return {k: [v] for (k, v) in self.items()}

    @property
    def dshape(self):
        return dshape_extract(self.as_row)

    # def last(self, *args, **kwargs):
    #    print("LAST")
    #    import pdb;pdb.set_trace()

    def key_of(self, id):
        """
        returns (key, status)
        key: the key associated to id
        status: {active|deleted}
        """
        if self._index is None:
            return list(self.keys())[id], 'active'
        if self._inverse is None:
            self._inverse = {i: k for (k, i) in self._index.items()}
        if id in self._inverse:
            return self._inverse[id], 'active'
        if self._inverse_del is None:
            self._inverse_del = {i: k for (k, i) in self._deleted.items()}
        if id in self._inverse_del:
            return self._inverse_del[id], 'deleted'
        raise KeyError(f"Key not found for id: {id}")

    def k_(self, id):
        k, _ = self.key_of(id)
        return k

    def fix_indices(self):  # TODO find a better name ...
        if self._index is None:
            return
        self._inverse = None
        self._inverse_del = None
        next_id = max(self.ids) + 1
        for k in self.keys():
            if k not in self._index:
                self._index[k] = next_id
                next_id += 1
            if k in self._deleted:  # a previously deleted key was added later
                del self._deleted[k]

    def created_indices(self, prev):
        if self._index is None:
            return bitmap(range(len(self))[len(prev):])
        new_keys = set(self.keys()) - set(prev.keys())
        return bitmap((i for (k, i) in self._index.items() if k in new_keys))

    def updated_indices(self, prev):
        if self._index is None:
            return bitmap((i for (i, x, y) in zip(range(len(prev)),
                                                  prev.values(),
                                                  self.values())
                           if x is not y))
        common_keys = set(self.keys()) & set(prev.keys())
        return bitmap((i for (k, i) in self._index.items()
                       if k in common_keys and self[k] is not prev[k]))

    def deleted_indices(self, prev):
        if self._index is None:
            return bitmap()
        del_keys = set(prev.keys()) - set(self.keys())
        return bitmap((i for (k, i) in self._deleted.items() if k in del_keys))

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(f"Key {key} does not exist")
        if self._index is None:  # first deletion
            self._index = dict(zip(self.keys(), range(len(self))))
        self._deleted[key] = self._index[key]
        del self._index[key]
        super().__delitem__(key)

    def clear(self):
        for key in list(self.keys()):
            del self[key]

    def set_nth(self, i, val):
        self[list(self)[i]] = val

    def get_nth(self, i):
        return self[list(self)[i]]

    @property
    def ids(self):
        if self._index is None:
            return bitmap(range(len(self)))
        return bitmap(self._index.values())
