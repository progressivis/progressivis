import numpy as np

class ProgressiveError(Exception):
    pass

NIL = np.array([],dtype=int)

def type_fullname(o):
    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__
    return module + '.' + o.__class__.__name__
