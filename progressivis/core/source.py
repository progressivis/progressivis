"Source allows keeping track of input sources and their progress"

from __future__ import annotations

import math

from typing import Any, Final, NamedTuple

UNKNOWN: Final[float] = math.nan
UNBOUND: Final[float] = math.inf


class Source(NamedTuple):
    "Base class for Module input sources."
    name: str
    data: Any
    length: float = UNKNOWN


class Progress(NamedTuple):
    "State of a the progression of a Source."
    source: Source
    cursor: float = 0
    end: float = UNKNOWN
