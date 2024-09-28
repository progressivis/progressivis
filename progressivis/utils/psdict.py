from __future__ import annotations

from ..core.pintset import PIntSet
from ..core.index_update import IndexUpdate
from ..table.dshape import DataShape, dshape_from_dict
import numpy as np


from typing import Any, List, Optional, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from progressivis.core.changemanager_dict import DictChangeManager


class PDict(Dict[str, Any]):
    "progressive dictionary"

    def __init__(self, other: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        if other is not None:
            # useful when keys are not varname-alike
            # one can use both (other and kwargs)
            # if keys are not duplicate
            assert isinstance(other, dict)
        else:
            other = {}
        super().__init__(other, **kwargs)
        self._index: Optional[Dict[str, int]] = None
        self._deleted: Dict[str, int] = {}
        self._inverse: Optional[Dict[int, str]] = None
        self._inverse_del: Optional[Dict[int, str]] = None
        self.changes: Optional[DictChangeManager] = None

    def compute_updates(
        self, start: int, now: float, mid: Optional[str], cleanup: bool = True
    ) -> Optional[IndexUpdate]:
        assert False, "compute_updates should not be called on PDict"
        if self.changes:
            updates = self.changes.compute_updates(start, now, mid, cleanup=cleanup)
            if updates is None:
                updates = IndexUpdate(created=PIntSet(self.ids))
            return updates
        return None

    def fill(self, val: Any) -> None:
        for k in self.keys():
            self[k] = val

    @property
    def array(self) -> np.ndarray[Any, Any]:
        return np.array(list(self.values()))

    @property
    def as_row(self) -> Dict[str, List[Any]]:
        return {k: [v] for (k, v) in self.items()}

    @property
    def dshape(self) -> DataShape:
        return dshape_from_dict(self.as_row)

    def key_of(self, id: int) -> Tuple[str, str]:
        """
        returns (key, status)
        key: the key associated to id
        status: {active|deleted}
        """
        if self._index is None:
            return list(self.keys())[id], "active"
        if self._inverse is None:
            self._inverse = {i: k for (k, i) in self._index.items()}
        if id in self._inverse:
            return self._inverse[id], "active"
        if self._inverse_del is None:
            self._inverse_del = {i: k for (k, i) in self._deleted.items()}
        if id in self._inverse_del:
            return self._inverse_del[id], "deleted"
        raise KeyError(f"Key not found for id: {id}")

    def k_(self, id: int) -> str:
        "Returns the key associated to id"
        k, _ = self.key_of(id)
        return k

    def fix_indices(self) -> None:  # TODO find a better name ...
        if self._index is None or not self.ids:
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

    def created_indices(self, prev: PDict) -> PIntSet:
        if self._index is None:
            index_ = dict(zip(self.keys(), range(len(self))))
        else:
            index_ = self._index
        new_keys = set(self.keys()) - set(prev.keys())
        return PIntSet((i for (k, i) in index_.items() if k in new_keys))

    def updated_indices(self, prev: PDict) -> PIntSet:
        if self._index is None:
            return PIntSet(
                (
                    i
                    for (i, x, y) in zip(range(len(prev)), prev.values(), self.values())
                    if x is not y
                )
            )
        common_keys = set(self.keys()) & set(prev.keys())
        return PIntSet(
            (
                i
                for (k, i) in self._index.items()
                if k in common_keys and self[k] is not prev[k]
            )
        )

    def deleted_indices(self, prev: PDict) -> PIntSet:
        if self._index is None:
            return PIntSet()
        del_keys = set(prev.keys()) - set(self.keys())
        return PIntSet((i for (k, i) in self._deleted.items() if k in del_keys))

    def __delitem__(self, key: str) -> None:
        if key not in self:
            raise KeyError(f"Key {key} does not exist")
        if self._index is None:  # first deletion
            self._index = dict(zip(self.keys(), range(len(self))))
        if key not in self._deleted:
            self._deleted[key] = self._index[key]
        if key in self._index:
            del self._index[key]
        super().__delitem__(key)

    def clear(self) -> None:
        for key in list(self.keys()):
            del self[key]

    def set_nth(self, i: int, val: Any) -> None:
        self[list(self)[i]] = val

    def get_nth(self, i: int) -> Any:
        return self[list(self)[i]]

    @property
    def ids(self) -> PIntSet:
        if self._index is None:
            return PIntSet(range(len(self)))
        return PIntSet(self._index.values())


EMPTY_PSDICT = PDict()
