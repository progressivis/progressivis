from __future__ import annotations

from abc import ABCMeta, abstractmethod, abstractproperty
from contextlib import contextmanager

from typing import (
    Callable,
    Union,
    Optional,
    Dict,
    Any,
    Tuple,
    TYPE_CHECKING,
    Iterable,
    Iterator,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    DTypeLike = npt.DTypeLike
    ArrayLike = npt.ArrayLike

Shape = Tuple[int, ...]


class StorageObject(metaclass=ABCMeta):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def attrs(self) -> Attribute:
        pass

    @abstractproperty
    def __len__(self) -> int:
        pass


class Attribute(metaclass=ABCMeta):
    @abstractmethod
    def __getitem__(self, name: str) -> Any:
        pass

    @abstractmethod
    def __setitem__(self, name: str, value: Any) -> None:
        pass

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __iter__(self) -> Iterable[str]:
        pass

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        pass


class DatasetFactory(StorageObject):
    @abstractmethod
    def create_dataset(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds: Any
    ) -> Dataset:
        pass

    @abstractmethod
    def require_dataset(
        self,
        name: str,
        shape: Shape,
        dtype: DTypeLike,
        exact: bool = False,
        **kwds: Any
    ) -> Dataset:
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> Any:
        pass

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        pass

    @abstractmethod
    def __contains__(self, name: str) -> bool:
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass


class Group(DatasetFactory):
    default: Optional[Callable[..., Group]] = None
    default_internal: Optional[Callable[[str], Group]] = None

    @abstractmethod
    def create_dataset(
        self,
        name: str,
        shape: Optional[Shape] = None,
        dtype: Optional[DTypeLike] = None,
        data: Optional[Any] = None,
        **kwds: Any
    ) -> Dataset:
        pass

    @abstractmethod
    def require_dataset(
        self,
        name: str,
        shape: Shape,
        dtype: DTypeLike,
        exact: bool = False,
        **kwds: Any
    ) -> Dataset:
        pass

    @abstractmethod
    def require_group(self, name: str) -> Group:
        pass

    @abstractmethod
    def __getitem__(self, name: str) -> Any:
        pass

    @abstractmethod
    def __delitem__(self, name: str) -> None:
        pass

    def close_all(self) -> None:
        pass

    def flush(self) -> None:
        pass


class Dataset(StorageObject):
    @abstractproperty
    def shape(self) -> Shape:
        pass

    @abstractproperty
    def dtype(self) -> np.dtype[Any]:
        pass

    @abstractproperty
    def maxshape(self) -> Shape:
        pass

    @abstractproperty
    def fillvalue(self) -> Any:
        pass

    @abstractproperty
    def chunks(self) -> Shape:
        pass

    @abstractmethod
    def resize(
        self, size: Union[int, Tuple[int, ...]], axis: Optional[int] = None
    ) -> None:
        pass

    @abstractproperty
    def size(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, args: Any) -> Any:
        pass

    @abstractmethod
    def __setitem__(self, args: Any, val: Any) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    def read_direct(
        self,
        dest: Any,
        source_sel: Optional[Union[int, slice]] = None,
        dest_sel: Optional[Union[int, slice]] = None,
    ) -> None:
        dest[dest_sel] = self[source_sel]


class StorageEngine(Group):
    _engines: Dict[str, StorageEngine] = dict()
    _default: Optional[str] = None

    def __init__(self, name: str, create_dataset_kwds: Optional[Dict[str, Any]] = None):
        # print('# creating storage engine %s'% name)
        # import pdb; pdb.set_trace()
        assert name not in StorageEngine._engines
        self._name = name
        StorageEngine._engines[name] = self
        if StorageEngine._default is None:
            StorageEngine._default = self.name
        self._create_dataset_kwds: Dict[str, Any] = dict(create_dataset_kwds or {})

    @staticmethod  # type: ignore
    @contextmanager
    def default_engine(engine: str) -> None:  # type: ignore
        if engine not in StorageEngine._engines:
            raise ValueError("Unknown storage engine %s", engine)
        saved = StorageEngine._default
        try:
            StorageEngine._default = engine
            yield saved
        finally:
            StorageEngine._default = saved

    @property
    def create_dataset_kwds(self) -> Dict[str, Any]:
        return self._create_dataset_kwds

    @property
    def name(self) -> str:
        return self._name

    def open(self, name: str, flags: Any, **kwds: Any) -> None:
        pass

    def close(self, name: str, flags: Any, **kwds: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    @staticmethod
    def lookup(engine: Optional[str]) -> Optional[StorageEngine]:
        assert StorageEngine._default is not None
        default = StorageEngine._engines.get(StorageEngine._default)
        if isinstance(engine, str):
            return StorageEngine._engines.get(engine, default)
        return default

    @staticmethod
    def engines() -> Dict[str, StorageEngine]:
        return StorageEngine._engines
