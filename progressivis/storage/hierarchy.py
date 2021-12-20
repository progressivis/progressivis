from __future__ import annotations

from typing import Union, Optional, Any, Dict, TYPE_CHECKING

from .base import Group, Attribute, Dataset

if TYPE_CHECKING:
    from .base import Shape, DTypeLike


class GroupImpl(Group):
    def __init__(self, name: str, parent: Optional[GroupImpl] = None):
        super(GroupImpl, self).__init__()
        self._name = name
        self.parent = parent
        self.dict: Dict[str, Union[Dataset, Group]] = {}
        self._attrs: Attribute = self._create_attribute()
        self._cparams = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def attrs(self) -> Attribute:
        return self._attrs

    def create_dataset(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds,
    ) -> Dataset:
        pass

    def _create_attribute(self, dict_values=None) -> Attribute:
        return AttributeImpl(dict_values)

    def require_dataset(
        self, name: str, shape: Shape, dtype: DTypeLike, exact=False, **kwds
    ) -> Dataset:
        _ = exact  # don't know what to do with it
        if name in self.dict:
            arr = self.dict[name]
        else:
            arr = self.create_dataset(name, shape, dtype, **kwds)
        assert isinstance(arr, Dataset)
        return arr

    def require_group(self, name: str) -> Group:
        group = self.get(name)
        if group is not None:
            assert isinstance(group, Group)
            return group
        return self.create_group(name, self)

    def _create_group(self, name: str, parent: Optional[GroupImpl]) -> Group:
        return GroupImpl(name, parent=parent)

    def create_group(self, item, overwrite=False) -> Group:
        path = normalize_storage_path(item)
        root = self
        if item and item[0] == "/":
            while root.parent is not None:
                root = root.parent
        path = path.split("/")
        for name in path[:-1]:
            value = root._get(name)
            if isinstance(value, GroupImpl):
                root = value
            else:
                raise ValueError(f"Path {item} contains a Dataset instead of a Group")
        name = path[-1]
        if name in self.dict and not overwrite:
            raise KeyError("%s already defined", item)
        group = self._create_group(name, self)
        self.dict[name] = group
        return group

    def _get(self, name: str, default=None) -> Union[Dataset, Group]:
        return self.dict.get(name, default)

    def get(
        self, item: str, default: Union[None, Dataset, Group] = None
    ) -> Union[None, Dataset, Group]:
        path = normalize_storage_path(item)
        if path == "":
            return self
        root = self
        if item and item[0] == "/":
            while root.parent is not None:
                root = root.parent
        path = path.split("/")
        ret: Union[None, Dataset, Group] = root
        for name in path:
            if isinstance(ret, GroupImpl):
                ret = ret.dict.get(name)
            else:
                raise ValueError(f"Path {item} is invalid")
        if ret is None:
            return default
        return ret

    def release(self, ids):
        pass

    def __getitem__(self, item: str) -> Union[Dataset, Group]:
        ret = self.get(item)
        if ret is None:
            raise KeyError("item named %s undefined", item)
        return ret

    def __delitem__(self, name: str) -> None:
        item = self.dict[name]
        self.free_item(item)
        del self.dict[name]

    def __contains__(self, name: str) -> bool:
        return name in self.dict

    def __len__(self) -> int:
        return len(self.dict)

    def free_item(self, item):
        pass


# Taken from zarr.util
def normalize_storage_path(path):
    if path:

        # convert backslash to forward slash
        path = path.replace("\\", "/")

        # ensure no leading slash
        while len(path) > 0 and path[0] == "/":
            path = path[1:]

        # ensure no trailing slash
        while len(path) > 0 and path[-1] == "/":
            path = path[:-1]

        # collapse any repeated slashes
        previous_char = None
        collapsed = ""
        for char in path:
            if char == "/" and previous_char == "/":
                pass
            else:
                collapsed += char
            previous_char = char
        path = collapsed

        # don't allow path segments with just '.' or '..'
        segments = path.split("/")
        if any([s in {".", ".."} for s in segments]):
            raise ValueError("path containing '.' or '..' segment not allowed")

    else:
        path = ""

    return path


class AttributeImpl(Attribute):
    def __init__(self, attrs_dict: Optional[Dict[str, Any]] = None):
        self.attrs: Dict[str, Any]
        if attrs_dict is None:
            self.attrs = dict()
        else:
            self.attrs = dict(attrs_dict)

    def __getitem__(self, name: str) -> Any:
        return self.attrs[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self.attrs[name] = value

    def __delitem__(self, name: str) -> None:
        del self.attrs[name]

    def __len__(self) -> int:
        return len(self.attrs)

    def __iter__(self):
        return iter(self.attrs)

    def __contains__(self, key: str) -> bool:
        return key in self.attrs

    def get(self, key, default=None) -> Optional[Any]:
        return self.attrs.get(key, default)
