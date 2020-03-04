from collections import defaultdict
from ..core.bitmap import bitmap

            
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
        
    def to_id(self, key):
        pass

    def to_key(self, id):
        pass

    def new_indices(self, prev):
        if self._index is None:
            return bitmap(range(len(self))[len(prev):])
        new_keys = set(self.keys()) - set(prev.keys())
        return bitmap((i for i in self.ids if self.to_key(i) in new_keys))

    def updated_indices(self, prev):
        if self._index is None:
            return bitmap((i for (i, x, y) in zip(range(len(prev)), prev.values(), self.values()) if x is not y))

    def deleted_indices(self, prev):
        if self._index is None:
            return bitmap()
        old_keys = set(prev.keys()) - set(self.keys())
        return bitmap((i for i in prev.ids if i not in self.ids))

    def __delitem__(self, _):
        raise NotImplementedError("key deletion not supported")

    
    def set_nth(self, i, val):
        self[list(self)[i]] = val
        
    def get_nth(self, i):
        return self[list(self)[i]]

    @property
    def ids(self):
        return range(len(self))
