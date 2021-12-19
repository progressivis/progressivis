from __future__ import annotations

from typing import Callable, Optional

from abc import ABCMeta, abstractmethod, abstractproperty
from contextlib import contextmanager


class StorageObject(metaclass=ABCMeta):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def attrs(self):
        pass

    @abstractproperty
    def __len__(self) -> int:
        pass


class Attribute(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, name):
        pass

    @abstractmethod
    def __setitem__(self, name, value):
        pass

    @abstractmethod
    def __delitem__(self, name):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __contains__(self, name):
        pass


class DatasetFactory(StorageObject):
    @abstractmethod
    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        pass

    @abstractmethod
    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        pass

    @abstractmethod
    def __getitem__(self, name):
        pass

    @abstractmethod
    def __delitem__(self, name):
        pass

    @abstractmethod
    def __contains__(self, name):
        pass


class Group(DatasetFactory):
    default: Optional[Callable[..., Group]] = None
    default_internal: Callable[[str], 'Group'] = None

    @abstractmethod
    def create_dataset(self, name, shape=None, dtype=None, data=None, **kwds):
        pass

    @abstractmethod
    def require_dataset(self, name, shape, dtype, exact=False, **kwds):
        pass

    @abstractmethod
    def require_group(self, name):
        pass

    @abstractmethod
    def __getitem__(self, name):
        pass

    @abstractmethod
    def __delitem__(self, name):
        pass

    @abstractmethod
    def __contains__(self, name):
        pass

    def close_all():
        pass


class Dataset(StorageObject):
    @abstractproperty
    def shape(self):
        pass

    @abstractproperty
    def dtype(self):
        pass

    @abstractproperty
    def maxshape(self):
        pass

    @abstractproperty
    def fillvalue(self):
        pass

    @abstractproperty
    def chunks(self):
        pass

    @abstractmethod
    def resize(self, size, axis=None):
        pass

    @abstractproperty
    def size(self):
        pass

    @abstractmethod
    def __getitem__(self, args):
        pass

    @abstractmethod
    def __setitem__(self, args, val):
        pass

    def read_direct(self, dest, source_sel=None, dest_sel=None):
        dest[dest_sel] = self[source_sel]


class StorageEngine(Group):
    _engines = dict()
    default = None

    def __init__(self, name, create_dataset_kwds=None):
        # print('# creating storage engine %s'% name)
        # import pdb; pdb.set_trace()
        assert name not in StorageEngine._engines
        self._name = name
        StorageEngine._engines[name] = self
        if StorageEngine.default is None:
            StorageEngine.default = self.name
        self._create_dataset_kwds = create_dataset_kwds or {}

    @staticmethod
    @contextmanager
    def default_engine(engine):
        if engine not in StorageEngine._engines:
            raise ValueError('Unknown storage engine %s', engine)
        saved = StorageEngine.default
        try:
            StorageEngine.default = engine
            yield saved
        finally:
            StorageEngine.default = saved

    @property
    def create_dataset_kwds(self):
        return self._create_dataset_kwds

    @property
    def name(self):
        return self._name

    def open(self, name, flags, **kwds):
        pass

    def close(self, name, flags, **kwds):
        pass

    def flush(self):
        pass

    @staticmethod
    def lookup(engine):
        default = StorageEngine._engines.get(StorageEngine.default)
        return StorageEngine._engines.get(engine, default)

    @staticmethod
    def engines():
        return StorageEngine._engines
