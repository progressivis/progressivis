from __future__ import absolute_import, division, print_function

from abc import abstractmethod
from .base import Group, Attribute, Dataset

class GroupImpl(Group):
    def __init__(self, name, parent=None):
        super(GroupImpl, self).__init__()
        self._name = name
        self.parent = parent
        self.dict = {}
        self._attrs = self._create_attribute()
        self._cparams = None

    @property
    def name(self):
        return self._name

    @property
    def attrs(self):
        return self._attrs

    @abstractmethod
    def create_dataset(self, name, shape=None, dtype=None, data=None, fillvalue=None, chunks=None, maxshape=None, **kwds):
        pass

    def _create_attribute(self, dict_values=None):
        return AttributeImpl(dict_values)


    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        _ = exact # don't know what to do with it
        if name in self.dict:
            arr = self.dict[name]
            assert(isinstance(arr, Dataset))
        else:
            arr = self.create_dataset(name, shape, dtype, **kwds)
        return arr

    def require_group(self, name):
        group = self.get(name)
        if group is not None:
            assert(isinstance(group, Group))
            return group
        return self.create_group(name, self)

    def _create_group(self, name, parent):
        return GroupImpl(name, parent=parent)

    def create_group(self, item, overwrite=False):
        path = normalize_storage_path(item)
        root = self
        if item and item[0] == '/':
            while root.parent is not None:
                root = root.parent
        path = path.split('/')
        for name in path[:-1]:
            root = root._get(name)
        name = path[-1]
        if name in self.dict and not overwrite:
            raise KeyError('%s already defined', item)
        group = self._create_group(name, self)
        self.dict[name] = group
        return group
        
    def _get(self, name, default=None):
        return self.dict.get(name, default)
    
    def get(self, item, default=None):
        path = normalize_storage_path(item)
        if path=='':
            return self
        root = self
        if item and item[0] == '/':
            while root.parent is not None:
                root = root.parent
        path = path.split('/')
        for name in path:
            root = root.dict.get(name)
            if root is None:
                return default
        return root

    def __getitem__(self, item):
        ret = self.get(item)
        if ret is None:
            raise KeyError('item named %s undefined', item)
        return ret

    def __delitem__(self, name):
        item = self.dict[name]
        self.free_item(item)
        del self.dict[name]

    def __contains__(self, name):
        return name in self.dict

    def __len__(self):
        return len(self.dict)

    def free_item(self, item):
        pass
            

# Taken from zarr.util
def normalize_storage_path(path):
    if path:

        # convert backslash to forward slash
        path = path.replace('\\', '/')

        # ensure no leading slash
        while len(path) > 0 and path[0] == '/':
            path = path[1:]

        # ensure no trailing slash
        while len(path) > 0 and path[-1] == '/':
            path = path[:-1]

        # collapse any repeated slashes
        previous_char = None
        collapsed = ''
        for char in path:
            if char == '/' and previous_char == '/':
                pass
            else:
                collapsed += char
            previous_char = char
        path = collapsed

        # don't allow path segments with just '.' or '..'
        segments = path.split('/')
        if any([s in {'.', '..'} for s in segments]):
            raise ValueError("path containing '.' or '..' segment not allowed")

    else:
        path = ''

    return path

class AttributeImpl(Attribute):
    def __init__(self, attrs_dict=None):
        if attrs_dict is None:
            self.attrs = dict()
        else:
            self.attrs = dict(attrs_dict)

    def __getitem__(self, name):
        return self.attrs[name]

    def __setitem__(self, name, value):
        self.attrs[name] = value

    def __delitem__(self, name):
        del self.attrs[name]

    def __len__(self):
        return len(self.attrs)

    def __iter__(self):
        return iter(self.attrs)

    def __contains__(self, key):
        return key in self.attrs

    def get(self, key, default=None):
        return self.attrs.get(key, default)
