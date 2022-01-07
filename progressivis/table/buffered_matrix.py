from __future__ import annotations

import logging

import numpy as np

from progressivis.core.utils import next_pow2


from typing import Any, Optional

logger = logging.getLogger(__name__)


class BufferedMatrix:
    def __init__(self) -> None:
        self._mat: Optional[np.ndarray[Any, Any]] = None
        self._base: Optional[np.ndarray[Any, Any]] = None

    def reset(self) -> None:
        self._mat = None
        self._base = None

    def matrix(self) -> Optional[np.ndarray[Any, Any]]:
        return self._mat

    def allocated_size(self) -> int:
        return 0 if self._base is None else int(self._base.shape[0])

    def __len__(self) -> int:
        return 0 if self._mat is None else int(self._mat.shape[0])

    def resize(self, newsize: int) -> np.ndarray[Any, Any]:
        lb = self.allocated_size()
        if newsize > lb:
            n = next_pow2(newsize)
            logger.info("Resizing matrix from %d to %d", lb, n)
            self._base = np.empty([n, n])  # defaults to float
            if self._mat is not None:
                # copy old content
                self._base[0 : self._mat.shape[0], 0 : self._mat.shape[1]] = self._mat
                logger.debug("Base matrix copied")
        assert self._base is not None
        self._mat = self._base[0:newsize, 0:newsize]
        assert self._mat is not None
        return self._mat
