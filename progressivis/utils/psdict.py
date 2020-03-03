from collections import defaultdict

class PKDict(dict):
    "permanent keys dictionary"
    def __init__(self, other=None, **kwargs):
        if other is not None:
            # useful when keys are not varname-alike
            # one can use both (other and kwargs)
            # if keys are not duplicate
            assert isinstance(other, dict)
        else:
            other = {}
        super().__init__(**other, **kwargs)
        self.activity = defaultdict(int)
        
    def __setitem__(self, key, val):
        super().__setitem__(key, val)
        self.activity[key] += 1

    def __delitem__(self, _):
        raise NotImplementedError("key deletion not supported")

    def reset_activity(self):
        self.activity = defaultdict(int)

class FKSDict(PKDict):
    "frozen key set dictionary"
    def __setitem__(self, key, val):
        if key not in self:
            raise KeyError(f"Unknown key '{key}'")
        super().__setitem__(key, val)
            
