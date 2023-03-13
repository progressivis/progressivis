"""
Isotropic Gaussian blobs
"""
from __future__ import annotations

import logging

import numpy as np
from abc import abstractmethod

from progressivis.core.module import ReturnRunStep
from ..utils.errors import ProgressiveError, ProgressiveStopIteration
from ..core.module import Module
from ..table.table import PTable
from ..table.dshape import dshape_from_dtype
from ..core.utils import integer_types
from ..core.module import def_output
from sklearn.datasets import make_blobs  # type: ignore
from sklearn.utils import shuffle as multi_shuffle  # type: ignore

from typing import Optional, Tuple, Any, List, Dict, Union, Callable, TypeVar
import numpy.typing as npt
T = TypeVar("T", bound=npt.NBitBase)

logger = logging.getLogger(__name__)

RESERVOIR_SIZE = 10000


def make_mv_blobs(
    means: List[float], covs: List[float], n_samples: int, **kwds: Any
) -> np.ndarray[Any, Any]:
    assert len(means) == len(covs)
    n_blobs = len(means)
    size = n_samples // n_blobs
    blobs = []
    labels = []
    for i, (mean, cov) in enumerate(zip(means, covs)):
        blobs.append(np.random.multivariate_normal(mean, cov, size, **kwds))
        arr = np.empty(size, dtype="int64")
        arr[:] = i
        labels.append(arr)
    blobs = np.concatenate(blobs)
    labels = np.concatenate(labels)
    return multi_shuffle(blobs, labels)  # type: ignore


def xy_to_dict(
    x: np.ndarray[Any, Any],
    y: np.ndarray[Any, Any],
    i: int,
    size: Optional[int],
    cols: Union[List[str], np.ndarray[Any, Any]],
) -> Tuple[Dict[str, Any], np.ndarray[Any, Any]]:
    res: Dict[str, Any] = {}
    k = None if size is None else i + size
    for j, col in enumerate(cols):
        res[col] = x[i:, j] if k is None else x[i:k, j]
    labs = y[i:] if k is None else y[i:k]
    return res, labs


@def_output("result", type=PTable)
@def_output("labels", type=PTable, required=False)
class BlobsPTableABC(Module):
    """Isotropic Gaussian blobs => table
    The purpose of the "reservoir" approach is to ensure the reproducibility of the results
    """

    kw_fun: Optional[Callable[..., Any]] = None

    def __init__(
        self,
        columns: Union[int, List[str], np.ndarray[Any, Any]],
        rows: int = -1,
        dtype: str = "float64",
        seed: int = 0,
        throttle: Union[int, bool, float] = False,
        **kwds: Any,
    ) -> None:
        super().__init__(**kwds)
        self.tags.add(self.TAG_SOURCE)
        dtype = dshape_from_dtype(np.dtype(dtype))
        self._kwds = {}  # FIXME
        """assert 'centers' in self._kwds
        assert 'n_samples' not in self._kwds
        assert 'n_features' not in self._kwds
        assert 'random_state' not in self._kwds"""
        # self._kwds['n_samples'] = rows
        # self._kwds['n_features']
        self.default_step_size = 1000
        self.columns: Union[List[str], np.ndarray[Any, Any]]
        if isinstance(columns, integer_types):
            self.columns = [f"_{i}" for i in range(1, columns + 1)]
            # self._kwds['n_features'] = columns
        elif isinstance(columns, (list, np.ndarray)):
            self.columns = columns
            # self._kwds['n_features'] = len(columns)
        else:
            raise ProgressiveError("Invalid type for columns")
        self.rows = rows
        self.seed = seed
        self._reservoir: Optional[
            Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
        ] = None
        self._reservoir_idx = 0
        if throttle and isinstance(throttle, integer_types + (float,)):
            self.throttle: Union[int, bool, float] = throttle
        else:
            self.throttle = False
        dshape = ", ".join([f"{col}: {dtype}" for col in self.columns])
        dshape = "{" + dshape + "}"
        table = PTable(self.generate_table_name("table"), dshape=dshape, create=True)
        self.result = table
        self.columns = table.columns

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("labels")
        if opt_slot:
            logger.debug("Maintaining labels")
            self.maintain_labels(True)
        else:
            logger.debug("Not maintaining labels")
            self.maintain_labels(False)

    def maintain_labels(self, yes: bool = True) -> None:
        if yes and self.labels is None:
            self.labels = PTable(
                self.generate_table_name("blobs_labels"),
                dshape="{labels: int64}",
                create=True,
            )
        elif not yes:
            self.labels = None

    @abstractmethod
    def fill_reservoir(self) -> None:
        pass

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        assert self.result is not None
        if step_size == 0:
            logger.error("Received a step_size of 0")
            return self._return_run_step(self.state_ready, steps_run=0)
        logger.info("generating %d lines", step_size)
        if self.throttle:
            step_size = np.min([self.throttle, step_size])
        if self.rows >= 0 and (len(self.result) + step_size) > self.rows:
            step_size = self.rows - len(self.result)
            logger.info("truncating to %d lines", step_size)
            if step_size <= 0:
                raise ProgressiveStopIteration
        if self._reservoir is None:
            self.fill_reservoir()
        steps = int(step_size)
        while steps > 0:
            assert self._reservoir
            level = len(self._reservoir[0]) - self._reservoir_idx
            assert level >= 0
            if steps >= level:
                blobs_dict, y_ = xy_to_dict(
                    *self._reservoir, self._reservoir_idx, None, self.columns
                )
                steps -= level
                # reservoir was emptied so:
                self.fill_reservoir()
            else:  # steps < level
                blobs_dict, y_ = xy_to_dict(
                    *self._reservoir, self._reservoir_idx, steps, self.columns
                )
                self._reservoir_idx += steps
                steps = 0
            self.result.append(blobs_dict)
            if self.labels is not None:
                self.labels.append({"labels": y_})
        if len(self.result) == self.rows:
            next_state = self.state_zombie
        elif self.throttle:
            next_state = self.state_blocked
        else:
            next_state = self.state_ready
        return self._return_run_step(next_state, steps_run=step_size)


class BlobsPTable(BlobsPTableABC):
    kw_fun = make_blobs

    def __init__(self, *args: Any, **kwds: Any) -> None:
        # import pdb;pdb.set_trace()
        super().__init__(*args, **kwds)
        self.centers = kwds.pop("centers")
        # assert 'centers' in self._kwds
        assert "n_samples" not in self._kwds
        assert "n_features" not in self._kwds
        assert "random_state" not in self._kwds

    def fill_reservoir(self) -> None:
        X, y = make_blobs(
            n_samples=RESERVOIR_SIZE,
            random_state=self.seed,
            centers=self.centers,
            **self._kwds,
        )
        self.seed += 1
        self._reservoir = (X, y)
        self._reservoir_idx = 0


class MVBlobsPTable(BlobsPTableABC):
    kw_fun = make_mv_blobs

    def __init__(self, *args: Any, **kwds: Any) -> None:
        super().__init__(*args, **kwds)
        self.means = kwds["means"]
        self.covs = kwds["covs"]

    def fill_reservoir(self) -> None:
        np.random.seed(self.seed)
        X, y = make_mv_blobs(
            n_samples=RESERVOIR_SIZE, means=self.means, covs=self.covs, **self._kwds
        )
        self.seed += 1
        self._reservoir = (X, y)
        self._reservoir_idx = 0
