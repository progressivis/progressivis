from __future__ import annotations

import numpy as np

from ..core.module import ReturnRunStep, JSon
from ..table.module import TableModule
from ..table import Table
from ..core.utils import indices_len
from progressivis import SlotDescriptor

try:
    from .knnkde import KNNKernelDensity
except Exception:
    pass

from typing import Optional, Any


class KernelDensity(TableModule):
    parameters = [
        ("samples", np.dtype(object), 1),
        ("bins", np.dtype(int), 1),
        ("threshold", np.dtype(int), 1000),
        ("knn", np.dtype(int), 100),
    ]
    inputs = [SlotDescriptor("table", type=Table, required=True)]

    def __init__(self, **kwds: Any) -> None:
        self._kde: Optional[KNNKernelDensity] = None
        self._json_cache: JSon = {}
        self._inserted: int = 0
        self._lately_inserted: int = 0
        super(KernelDensity, self).__init__(**kwds)
        self.tags.add(self.TAG_VISUALIZATION)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("table")
        assert dfslot is not None
        if dfslot.deleted.any():
            raise ValueError("Not implemented yet")
        if not dfslot.created.any():
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.created.next(length=step_size, as_slice=False)
        steps = indices_len(indices)
        if steps == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self._kde is None:
            self._kde = KNNKernelDensity(dfslot.data(), online=True)
        res = self._kde.run_ids(indices.to_array())
        self._inserted += res["numPointsInserted"]
        self._lately_inserted += steps
        samples = self.params.samples
        sample_num = self.params.bins
        threshold = self.params.threshold
        knn = self.params.knn
        if self._lately_inserted > threshold:
            scores = self._kde.score_samples(samples.astype(np.float32), k=knn)  # type: ignore
            self._lately_inserted = 0
            self._json_cache = {
                "points": np.array(
                    dfslot.data().loc[:500, :].to_dict(orient="split")["data"]
                ).tolist(),
                "bins": sample_num,
                "inserted": self._inserted,
                "total": len(dfslot.data()),
                "samples": [
                    (sample, score)
                    for sample, score in zip(samples.tolist(), scores.tolist())  # type: ignore
                ],
            }
        return self._return_run_step(self.state_ready, steps_run=steps)

    def get_visualization(self) -> Optional[str]:
        return "knnkde"

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super(KernelDensity, self).to_json(short, with_speed)
        if short:
            return json
        return self.knnkde_to_json(json)

    def knnkde_to_json(self, json: JSon) -> JSon:
        json.update(self._json_cache)
        return json
