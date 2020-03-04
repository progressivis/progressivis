from collections import defaultdict
from ..core.bitmap import bitmap

SECOND_BEGIN = 2 ** 12
            
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
        super().__init__(**other, **kwargs)
        self._index = None
        self._cnt = SECOND_BEGIN
        
    def to_id(self, key):
        pass

    def to_key(self, id):
        pass

    def fix_indices(self): # TODO find a better name ...
        if self._index is None:
            return
        for k in self.keys():
            if k not in self._index:
                self._index[k] = self._cnt
                self._cnt += 1
    
    def new_indices(self, prev):
        if self._index is None:
            return bitmap(range(len(self))[len(prev):])
        new_keys = set(self.keys()) - set(prev.keys())
        #self.fix_indices()
        return bitmap((i for (k, i) in self._index.items() if k in new_keys))

    def updated_indices(self, prev):
        if self._index is None:
            return bitmap((i for (i, x, y) in zip(range(len(prev)), prev.values(), self.values()) if x is not y))
        common_keys = set(self.keys()) & set(prev.keys())
        #self.fix_indices()
        return bitmap((i for (k, i) in  self._index.items() if k in common_keys and self[k] is not prev[k]))

    def deleted_indices(self, prev):
        if self._index is None:
            return bitmap()
        del_keys = set(prev.keys()) - set(self.keys())
        return bitmap((i[0] for (k, i) in self._index.items() if k in del_keys and isinstance(i, tuple)))

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(f"Key {key} does not exist")
        if self._index is None: # first deletion
            self._index = dict(zip(self.keys(), range(len(self))))
        self._index[key] = (self._index[key],) # tuple([i]) => deletion mark
        super().__delitem__(key)
    
    def set_nth(self, i, val):
        self[list(self)[i]] = val
        
    def get_nth(self, i):
        return self[list(self)[i]]

    @property
    def ids(self):
        if self._index is None:
            return range(len(self))
        #self.fix_indices()
        return [i for i in self._index.values() if isinstance(i, int)]
