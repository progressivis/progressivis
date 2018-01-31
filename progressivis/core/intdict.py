from __future__ import absolute_import, division, print_function

from progressivis.core.utils import integer_types
import numpy as np
import six
if six.PY2:
    range = xrange


try:
    from progressivis.core.khash.hashtable import Int64HashTable

    class IntDict(object):
        def __init__(self, keys=None, values=None, **kwargs):
            if keys is None:
                assert values is None or len(values) == 0
                self._ht = Int64HashTable(0)
            else:
                l = len(keys)
                assert l == len(values)
                self._ht = Int64HashTable(l)
                self._ht.map(keys, np.asarray(values, dtype=np.int64))

        def __getitem__(self, key):
            return self._ht.get_item(key)

        def __setitem__(self, key, val):
            assert(isinstance(key, integer_types) and
                   isinstance(val, integer_types))
            self._ht.set_item(key, val)

        def __delitem__(self, key):
            self._ht.del_item(key)

        def get_values(self, values):
            return self._ht.get_items(values)

        def update(self, keys, values):
            self._ht.map(keys, np.asarray(values, dtype=np.int64))

        def get_items(self, key_value):
            assert (isinstance(key_value, np.ndarray) and
                    key_value.dtype == np.int64)
            self._ht.get_items(key_value)
            return key_value

        def __contains__(self, key):
            return key in self._ht

        def __len__(self):
            return len(self._ht)

        def contains_any(self, keys):
            assert isinstance(keys, np.ndarray) and keys.dtype == np.int64
            return self._ht.contains_any(keys)

except ImportError:
    print("# Cannot load IntHashTable")

    class IntDict(dict):
        def __init__(self, *args, **kwargs):
            if len(args) == 2:
                dict.__init__(self, zip(args[0], args[1]), **kwargs)
            else:
                dict.__init__(self, *args, **kwargs)

        def __getitem__(self, key):
            val = dict.__getitem__(self, key)
            return val

        def __setitem__(self, key, val):
            assert type(key) == int and type(val) == int
            dict.__setitem__(self, key, val)

        def update(self, keys, values):
            dict.update(self, zip(keys, values))

        def get_items(self, key_value):
            assert (isinstance(key_value, np.ndarray) and
                    key_value.dtype == np.int64)
            for i in range(len(key_value)):
                key_value[i] = self[key_value[i]]
            return key_value

        def contains_any(self, keys):
            assert isinstance(keys, np.ndarray) and keys.dtype == np.int64
            for k in keys:
                if k in self:
                    return True
            return False
