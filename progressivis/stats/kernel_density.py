from __future__ import annotations

import numpy as np

from ..core.module import Module, ReturnRunStep, JSon, def_input, def_parameter
from ..table.api import PTable
from ..core.utils import indices_len

try:
    from .knnkde import KNNKernelDensity
except Exception:
    pass

from typing import Any


@def_parameter("samples", np.dtype(object), 1)
@def_parameter("bins", np.dtype(int), 1)
@def_parameter("threshold", np.dtype(int), 1000)
@def_parameter("knn", np.dtype(int), 100)
@def_input("table", PTable, hint_type=dict[str, str])
# @def_output("result", PTable)
class KernelDensity(Module):
    def __init__(
        self,
        x_column: int | str = "",
        y_column: int | str = "",
        **kwds: Any
    ) -> None:
        self.x_column = x_column
        self.y_column = y_column
        self._kde: KNNKernelDensity | None = None
        self._json_cache: JSon = {}
        self._inserted: int = 0
        self._lately_inserted: int = 0
        super().__init__(**kwds)
        self.tags.add(self.TAG_VISUALIZATION)

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("table")
        assert dfslot is not None
        if self.x_column == "":
            assert self.y_column == ""
            assert dfslot.hint is not None
            assert len(dfslot.hint) == 2
            self.x_column = dfslot.hint["x"]
            self.y_column = dfslot.hint["y"]
        if dfslot.deleted.any():
            raise ValueError("Not implemented yet")

        if not dfslot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.created.next(length=step_size, as_slice=False)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._kde is None:
            self._kde = KNNKernelDensity(dfslot.data().loc[:, [self.x_column, self.y_column]], online=True)
        res = self._kde.run_ids(indices.to_array())
        self._inserted += res["numPointsInserted"]
        self._lately_inserted += steps
        samples = self.params.samples
        sample_num = self.params.bins
        threshold = self.params.threshold
        knn = self.params.knn
        if self._lately_inserted > threshold:
            scores = self._kde.score_samples(samples.astype(np.float32), k=knn)
            self._lately_inserted = 0
            self._json_cache = {
                "points": np.array(
                    dfslot.data().loc[np.random.choice(indices, 500),  # type: ignore
                                      [self.x_column,
                                       self.y_column]].to_dict(orient="split")["data"]
                ).tolist(),
                "bins": sample_num,
                "inserted": self._inserted,
                "total": len(dfslot.data()),
                "samples": [
                    (sample, score)
                    for sample, score in zip(samples, scores)  # type: ignore
                ],
            }
        return self._return_run_step(self.state_ready, steps_run=steps)

    def get_visualization(self) -> str:
        return "knnkde"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short:
            return json
        return self.knnkde_to_json(json)

    def knnkde_to_json(self, json: JSon) -> JSon:
        json.update(self._json_cache)
        return json
