from __future__ import annotations

import logging

from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # type: ignore
from sklearn.utils.validation import check_random_state  # type: ignore
from progressivis import ProgressiveError
from progressivis.core.module import (
    ReturnRunStep,
    JSon,
    def_input,
    def_output,
    def_parameter,
)
from progressivis.core.pintset import PIntSet
from progressivis.core.utils import indices_len
from ..core.module import Module
from ..core.decorators import process_slot, run_if_any
from ..core.quality import QualitySqrtSumSquarredDiffs
from ..table.api import PTable, PTableSelectedView
from ..table.dshape import dshape_from_dtype, dshape_from_columns
from progressivis.io.api import Variable
from ..utils.psdict import PDict
from ..table.filtermod import FilterMod
from ..stats.api import Var

from typing import Optional, Union, List, Dict, Any, cast

logger = logging.getLogger(__name__)


@def_parameter("samples", np.dtype(int), 50)
@def_input("table", PTable)
@def_input("var", PTable)
@def_input("moved_center", type=PDict, required=False)
@def_output("result", PTable)
@def_output("labels", type=PTable, attr_name="_labels", required=False)
@def_output("nz_labels", type=PIntSet, required=False)
@def_output("label_dict", type=PDict, required=False)
@def_output("conv", type=PDict, attr_name="_conv_out", required=False)
class MBKMeans(Module):
    """
    Mini-batch k-means using the sklearn implementation.
    """

    def __init__(
        self,
        n_clusters: int,
        batch_size: int = 100,
        tol: float = 0.01,
        is_input: bool = True,
        is_greedy: bool = True,
        random_state: Union[int, np.random.RandomState, None] = None,
        **kwds: Any,
    ):
        super().__init__(**kwds)
        self.mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            verbose=0,
            tol=tol,
            random_state=random_state,
        )
        self.n_clusters = n_clusters
        self.default_step_size = 100
        self._remaining_inits = 10
        self._initialization_steps = 0
        self._is_input = is_input
        self._tol = tol
        self._conv_out = PDict({"convergence": "unknown"})
        self.params.samples = n_clusters
        self._is_greedy: bool = is_greedy
        self._has_converged: bool = False
        self._arrays: Optional[Dict[int, np.ndarray[Any, Any]]] = None
        self._quality: QualitySqrtSumSquarredDiffs | None = None
        # self.convergence_context = {}

    def reset(self, init: str | None = None) -> None:
        self.reset_mbk()
        dfslot = self.get_input_slot("table")
        dfslot.reset()
        self.set_state(self.state_ready)
        # do not resize result to zero
        #if self._labels is not None:
        #    self._labels.truncate()

    def reset_mbk(self, init: str | None = None) -> None:
        if init is None:
            init = self.mbk.cluster_centers_ if hasattr(self.mbk, "cluster_centers_") else "k-means++"
        self.mbk = MiniBatchKMeans(
            n_clusters=self.mbk.n_clusters,
            batch_size=self.mbk.batch_size,
            init=init,
            random_state=self.mbk.random_state,
        )

    def starting(self) -> None:
        super().starting()
        if self.get_output_slot("labels"):
            logger.debug("Maintaining labels")
            self.maintain_labels(True)
        else:
            logger.debug("Not maintaining labels")
            self.maintain_labels(False)
        if self.get_output_slot("label_dict"):
            self.label_dict = PDict({i: PIntSet() for i in range(1, self.n_clusters+1)})  # type: ignore
        if self.get_output_slot("nz_labels"):
            self.nz_labels = PIntSet()

    def maintain_labels(self, yes: bool = True) -> None:
        if yes and self._labels is None:
            self._labels = PTable(
                self.generate_table_name("labels"),
                dshape="{label: int32, run_number: int32}",
                create=True,
            )
        elif not yes:
            self._labels = None

    def is_greedy(self) -> bool:
        return self._is_greedy
    def _process_labels(self, locs: PIntSet, run_number: int, labels: np.ndarray[Any, Any] | None = None) -> None:
        if labels is not None:
            labels += 1
        else:
            labels = self.mbk.labels_ + 1
        assert self._labels is not None
        self._labels["label"].loc[locs] = labels
        self._labels["run_number"].loc[locs] = run_number
        if self.nz_labels is not None:
            assert isinstance(locs, PIntSet)
            self.nz_labels.update(locs)

    def _process_label_dict(self, locs: PIntSet, run_number: int, labels: np.ndarray[Any, Any] | None = None) -> None:
        if labels is not None:
            labels += 1
        else:
            labels = self.mbk.labels_ + 1
        assert self.label_dict is not None
        for i, ix in enumerate(locs):
            self.label_dict[labels[i]].add(ix)
        self.label_dict[0] = run_number  # type: ignore

    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("table")
        # TODO varslot is only required if we have tol > 0
        varslot = self.get_input_slot("var")
        if dfslot.deleted.any() or dfslot.updated.any():
            logger.debug("has deleted or updated, reseting")
            varslot.reset()
            self.reset()
        if dfslot.created.any():
            self._has_converged = False
            self.reset_mbk()
            # print("-", end="")
        dfslot.clear_buffers()
        varslot.clear_buffers()
        moved_center = self.get_input_slot("moved_center") if self.has_input_slot("moved_center") else None
        if moved_center is not None:
            if moved_center.has_buffered():
                print("Moved center!!")
                self._has_converged = False
                print("No convergence move center")
                moved_center.clear_buffers()
                msg = moved_center.data()
                for c in msg:
                    self.set_centroid(c, msg[c][:2])
                self.mbk._counts.fill(0)
                self.reset()
                dfslot.clear_buffers()  # No need to re-reset next
                varslot.clear_buffers()
        if self._has_converged:
            return self.run_step_labels(run_number, step_size, quantum)
        # print('dfslot has buffered %d elements'% dfslot.created_length())
        input_df = dfslot.data()
        var_data = varslot.data()
        batch_size = self.mbk.batch_size or 100
        if (
            input_df is None
            or var_data is None
            or len(input_df) < max(self.mbk.n_clusters, batch_size)
        ):
            # Not enough data yet ...
            return self._return_run_step(self.state_blocked, 0)
        cols = dfslot.hint or input_df.columns
        dtype = input_df.columns_common_dtype(cols)
        n_features = len(cols)
        n_samples = len(input_df)
        if self._arrays is None:

            def _array_factory() -> np.ndarray[Any, Any]:
                return np.empty((self._key, n_features), dtype=dtype)

            self._arrays = defaultdict(_array_factory)
        is_conv = False
        if self._tol > 0:
            # v = np.array(list(var_data.values()), dtype=np.float64)
            # tol = np.mean(v) * self._tol
            prev_centers = np.zeros((self.n_clusters, n_features), dtype=dtype)
        else:
            # tol = 0
            prev_centers = np.zeros(0, dtype=dtype)
        random_state = check_random_state(self.mbk.random_state)
        X: Optional[np.ndarray[Any, Any]] = None
        # Attributes to monitor the convergence
        self.mbk._ewa_inertia = None
        self.mbk._ewa_inertia_min = None
        self.mbk._no_improvement = 0
        for iter_ in range(step_size):
            mb_ilocs = random_state.randint(0, n_samples, batch_size)
            mb_locs = input_df.index[mb_ilocs]  # sorts and removes duplicates
            self._key = len(mb_locs)
            arr = self._arrays[self._key]  # reuse previously created arrays rather than create new ones at each step
            X = input_df.to_array(columns=cols, locs=mb_locs, ret=arr)
            if hasattr(self.mbk, "cluster_centers_"):
                prev_centers[:, :] = self.mbk.cluster_centers_
            self.mbk.partial_fit(X)
            if self._labels is not None:
                self._labels.resize(len(input_df))
                self._process_labels(mb_locs, run_number)
            if self.label_dict is not None:
                self._process_label_dict(mb_locs, run_number)
            centers = self.mbk.cluster_centers_
            nearest_center, batch_inertia = self.mbk.labels_, self.mbk.inertia_
            k = centers.shape[0]
            squared_diff = 0.0
            for ci in range(k):
                center_mask = nearest_center == ci
                if np.count_nonzero(center_mask) > 0:
                    diff = centers[ci].ravel() - prev_centers[ci].ravel()
                    squared_diff += np.dot(diff, diff)
            # 3 necessary settings before calling _mini_batch_convergence
            self.mbk._ewa_inertia = None
            self._no_improvement = 0
            self.mbk._counts.fill(0)
            if self.mbk._mini_batch_convergence(
                iter_, step_size, n_samples, squared_diff, batch_inertia
            ):
                is_conv = True
                break
        if self.result is None:
            assert X is not None
            dshape = dshape_from_columns(input_df, cols, dshape_from_dtype(X.dtype))
            self.result = PTable(
                self.generate_table_name("centers"), dshape=dshape, create=True
            )
            self.result.resize(self.mbk.cluster_centers_.shape[0])
        self.result[cols] = self.mbk.cluster_centers_
        if is_conv:
            self._has_converged = True
            # print("+", end="")
            if self._labels is None and self.label_dict is None:
                return self._return_run_step(self.state_blocked, iter_)
            dfslot.reset()
            self._is_greedy = False
        return self._return_run_step(self.state_ready, iter_)

    def run_step_labels(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        print("fix labels")
        varslot = self.get_input_slot("var")
        varslot.clear_buffers()
        dfslot = self.get_input_slot("table")
        assert dfslot is not None
        if dfslot.deleted.any() or dfslot.updated.any():
            dfslot.reset()
        indices = dfslot.created.next(length=step_size, as_slice=False)
        steps = len(indices)
        if not steps:
            return self._return_run_step(self.next_state(dfslot), steps_run=steps)
        input_df = dfslot.data()
        cols = dfslot.hint or input_df.columns
        assert input_df is not None
        X = input_df.to_array(columns=cols, locs=indices)  # TODO: improve to_atrray() for slices
        labels = self.mbk.predict(X)
        if self._labels is not None:
            self._labels.resize(len(input_df))
            self._process_labels(indices, run_number, labels)
        if self.label_dict is not None:
            for k, v in self.label_dict.items():
                if k == 0:  # type: ignore
                    continue
                v.difference_update(indices)
            self._process_label_dict(indices, run_number, labels)
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short or self.result is None:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json: JSon) -> JSon:
        assert self.result is not None
        json["cluster_centers"] = self.result.to_json()
        return json

    def set_centroid(self, c: int, values: List[float]) -> List[float]:
        try:
            c = int(c)
        except ValueError:
            pass
        assert self.result is not None
        centroids = self.result
        # idx = centroids.id_to_index(c)

        dfslot = self.get_input_slot("table")
        input_df = dfslot.data()
        columns = dfslot.hint or input_df.columns
        if len(values) != len(columns):
            raise ProgressiveError(f"Expected {len(columns)} values, received {values}")
        centroids.loc[c, columns] = values
        # TODO unpack the table
        centers = centroids.loc[c, columns]
        self.mbk.cluster_centers_[c] = list(centers)
        return cast(List[float], self.mbk.cluster_centers_.tolist())

    def create_dependent_modules(
        self, input_module: Module, input_slot: str = "result"
    ) -> None:
        with self.grouped():
            s = self.scheduler
            self.input_module = input_module
            self.input.table = input_module.output[input_slot]
            self.input_slot = input_slot
            c = Variable(group=self.name, scheduler=s)
            self.dep.moved_center = c
            self.input.moved_center = c.output.result
            v = Var(group=self.name, scheduler=s)
            self.dep.variance = v
            v.input.table = input_module.output[input_slot]
            self.input.var = v.output.result

    def get_quality(self) -> Dict[str, float]:
        if self._quality is None:
            self._quality = QualitySqrtSumSquarredDiffs()
        if hasattr(self.mbk, "cluster_centers_"):  # attr does not exist before a fit
            return {"mb_k_means": self._quality.quality(self.mbk.cluster_centers_)}
        return {"mb_k_means": 0}


@def_input("table", PTable)
@def_input("labels", PTable)
@def_output("result", PTableSelectedView)
class MBKMeansFilter(Module):
    """
    Filters data corresponding to a specific label
    """

    def __init__(self, sel: Any, **kwds: Any) -> None:
        self._sel = sel+1
        super().__init__(**kwds)

    @process_slot("table", "labels")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices_t = ctx.table.created.next(length=step_size)  # returns a slice
            steps_t = indices_len(indices_t)
            ctx.table.clear_buffers()
            indices_l = ctx.labels.created.next(length=step_size)  # returns a slice
            steps_l = indices_len(indices_l)
            ctx.labels.clear_buffers()
            steps = steps_t + steps_l
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.result is None:
                self.result = PTableSelectedView(
                    ctx.table.data(), ctx.labels.data().selection
                )
            else:
                self.result.selection = ctx.labels.data().selection
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def create_dependent_modules(
        self, mbkmeans: MBKMeans, data_module: Module, data_slot: str
    ) -> None:
        with self.grouped():
            scheduler = self.scheduler
            filter_ = FilterMod(expr=f"label=={self._sel}", scheduler=scheduler)
            filter_.input.table = mbkmeans.output.labels
            filter_.input.selection = mbkmeans.output.nz_labels
            self.dep.filter = filter_
            self.input.labels = filter_.output.result
            self.input.table = data_module.output[data_slot]

@def_input("table", PTable)
@def_input("label_dict", PDict)
@def_output("result", PTableSelectedView)
class MBKMeansSelector(Module):
    """
    Filters data corresponding to a specific label
    """

    def __init__(self, sel: Any, **kwds: Any) -> None:
        self._sel = sel+1
        super().__init__(**kwds)

    @process_slot("table", "label_dict")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, quantum: float
    ) -> ReturnRunStep:
        assert self.context
        with self.context as ctx:
            indices_t = ctx.table.created.next(length=step_size)  # returns a slice
            steps = indices_len(indices_t)
            ctx.table.clear_buffers()
            ctx.label_dict.clear_buffers()
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.result is None:
                self.result = PTableSelectedView(
                    ctx.table.data(), ctx.label_dict.data()[self._sel]
                )
            else:
                self.result.selection = ctx.label_dict.data()[self._sel]
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)
