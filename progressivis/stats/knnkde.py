# Author: Jaemin Jo <jmjo@hcil.snu.ac.kr>
from __future__ import annotations


import numpy as np

try:
    from pynene import Index  # type: ignore
except Exception:
    pass


from typing import Optional, Any, Iterable


class KNNKernelDensity:
    SQRT2PI = np.sqrt(2 * np.pi)

    def __init__(self, X: np.ndarray[Any, Any], online: Optional[bool] = False):
        self.X = X
        self.index = Index(X)
        if not online:
            self.index.add_points(len(X))

    def run(self, ops: Any) -> Any:
        return self.index.run(ops)

    def run_ids(self, ids: Iterable[int]) -> Any:
        return self.index.run_ids(ids)

    def score_samples(
        self, X: np.ndarray[Any, Any], k: int = 10, bandwidth: float = 0.2
    ) -> float:
        _, dists = self.index.knn_search_points(X, k=k)
        scores = self._gaussian_score(dists, bandwidth) / k
        return scores

    def _gaussian_score(self, dists: float, bandwidth: float) -> float:
        logg = -0.5 * (dists / bandwidth) ** 2
        g = np.exp(logg) / bandwidth / self.SQRT2PI
        return g.sum(axis=1)  # type: ignore
