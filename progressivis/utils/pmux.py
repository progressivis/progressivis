from typing import Any, Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .psdict import PDict


class PMux:
    def __init__(self) -> None:
        self._results: Dict[Tuple[str, str], PDict] = {}

    def update(self, key: Tuple[str, str], value: Dict[str, Any]) -> None:
        from .psdict import PDict
        if key not in self._results:
            self._results[key] = PDict()
        self._results[key].update(value)

    def clear(self) -> None:
        for d in self._results.values():
            d.clear()
