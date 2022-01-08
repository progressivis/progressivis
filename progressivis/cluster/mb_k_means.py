from __future__ import annotations

import logging

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans  # type: ignore
from sklearn.utils.validation import check_random_state  # type: ignore
from progressivis import ProgressiveError, SlotDescriptor
from progressivis.core.utils import indices_len
from ..table.module import TableModule, ReturnRunStep, JSon, Module
from ..table.table_base import BaseTable
from ..table import Table, TableSelectedView
from ..table.dshape import dshape_from_dtype, dshape_from_columns
from ..io import DynVar
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from ..table.filtermod import FilterMod
from ..stats import Var

from typing import Optional, Union, List, Dict, Any

logger = logging.getLogger(__name__)


class MBKMeans(TableModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """

    parameters = [("samples", np.dtype(int), 50)]
    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("var", type=Table, required=True),
        SlotDescriptor("moved_center", type=PsDict, required=False),
    ]
    outputs = [
        SlotDescriptor("labels", type=Table, required=False),
        SlotDescriptor("conv", type=PsDict, required=False),
    ]

    def __init__(
        self,
        n_clusters: int,
        columns: List[str] = None,
        batch_size: int = 100,
        tol: float = 0.01,
        is_input=True,
        is_greedy=True,
        random_state: Union[int, np.random.RandomState, None] = None,
        **kwds,
    ):
        super().__init__(columns=columns, **kwds)
        self.mbk = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=batch_size,
            verbose=True,
            tol=tol,
            random_state=random_state,
        )
        self.n_clusters = n_clusters
        self.default_step_size = 100
        self._labels: Optional[Table] = None
        self._remaining_inits = 10
        self._initialization_steps = 0
        self._is_input = is_input
        self._tol = tol
        self._conv_out = PsDict({"convergence": "unknown"})
        self.params.samples = n_clusters
        self._is_greedy = is_greedy
        self._arrays: Optional[Dict[int, np.ndarray]] = None
        # self.convergence_context = {}

    def predict_step_size(self, duration: float) -> int:
        p = super().predict_step_size(duration)
        return max(p, self.n_clusters)

    def reset(self, init="k-means++") -> None:
        self.mbk = MiniBatchKMeans(
            n_clusters=self.mbk.n_clusters,
            batch_size=self.mbk.batch_size,
            init=init,
            random_state=self.mbk.random_state,
        )
        dfslot = self.get_input_slot("table")
        dfslot.reset()
        self.set_state(self.state_ready)
        # self.convergence_context = {}
        # do not resize result to zero
        # it contains 1 row per centroid
        if self._labels is not None:
            self._labels.truncate()

    def starting(self) -> None:
        super().starting()
        opt_slot = self.get_output_slot("labels")
        if opt_slot:
            logger.debug("Maintaining labels")
            self.maintain_labels(True)
        else:
            logger.debug("Not maintaining labels")
            self.maintain_labels(False)

    def maintain_labels(self, yes=True) -> None:
        if yes and self._labels is None:
            self._labels = Table(
                self.generate_table_name("labels"),
                dshape="{labels: int64}",
                create=True,
            )
        elif not yes:
            self._labels = None

    def labels(self) -> Optional[Table]:
        return self._labels

    def get_data(self, name: str) -> Any:
        if name == "labels":
            return self.labels()
        if name == "conv":
            return self._conv_out
        return super().get_data(name)

    def is_greedy(self) -> bool:
        return self._is_greedy

    def _process_labels(self, locs):
        labels = self.mbk.labels_
        u_locs = locs & self._labels.index  # ids to update
        if not u_locs:  # shortcut
            self._labels.append({"labels": labels}, indices=locs)
            return
        a_locs = locs - u_locs  # ids to append
        if not a_locs:  # 2nd shortcut
            self._labels.loc[locs, "labels"] = labels
            return
        df = pd.DataFrame({"labels": labels}, index=locs)
        u_labels = df.loc[u_locs, "labels"]
        a_labels = df.loc[a_locs, "labels"]
        self._labels.loc[u_locs, "labels"] = u_labels
        self._labels.append({"labels": a_labels}, indices=a_locs)

    def run_step(
        self, run_number: int, step_size: int, howlong: float
    ) -> ReturnRunStep:
        dfslot = self.get_input_slot("table")
        # TODO varslot is only required if we have tol > 0
        varslot = self.get_input_slot("var")
        moved_center = self.get_input_slot("moved_center")
        init_centers = "k-means++"
        if moved_center is not None:
            if moved_center.has_buffered():
                print("Moved center!!")
                moved_center.clear_buffers()
                msg = moved_center.data()
                for c in msg:
                    self.set_centroid(c, msg[c][:2])
                init_centers = self.mbk.cluster_centers_
                self.reset(init=init_centers)
                dfslot.clear_buffers()  # No need to re-reset next
                varslot.clear_buffers()
        if dfslot.has_buffered() or varslot.has_buffered():
            logger.debug("has deleted or updated, reseting")
            self.reset(init=init_centers)
            dfslot.clear_buffers()
            varslot.clear_buffers()
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
        cols = self.get_columns(input_df, "table")
        dtype = input_df.columns_common_dtype(cols)
        n_features = len(cols)
        n_samples = len(input_df)
        if self._arrays is None:

            def _array_factory():
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
        X: Optional[np.ndarray] = None
        # Attributes to monitor the convergence
        self.mbk._ewa_inertia = None
        self.mbk._ewa_inertia_min = None
        self.mbk._no_improvement = 0
        for iter_ in range(step_size):
            mb_ilocs = random_state.randint(0, n_samples, batch_size)
            mb_locs = input_df.index[mb_ilocs]
            self._key = len(mb_locs)
            arr = self._arrays[self._key]
            X = input_df.to_array(columns=cols, locs=mb_locs, ret=arr)
            if hasattr(self.mbk, "cluster_centers_"):
                prev_centers[:, :] = self.mbk.cluster_centers_
            self.mbk.partial_fit(X)
            if self._labels is not None:
                self._process_labels(mb_locs)
            centers = self.mbk.cluster_centers_
            nearest_center, batch_inertia = self.mbk.labels_, self.mbk.inertia_
            k = centers.shape[0]
            squared_diff = 0.0
            for ci in range(k):
                center_mask = nearest_center == ci
                if np.count_nonzero(center_mask) > 0:
                    diff = centers[ci].ravel() - prev_centers[ci].ravel()
                    squared_diff += np.dot(diff, diff)
            if self.mbk._mini_batch_convergence(
                iter_, step_size, n_samples, squared_diff, batch_inertia
            ):
                is_conv = True
                break
        if self.result is None:
            assert X is not None
            dshape = dshape_from_columns(input_df, cols, dshape_from_dtype(X.dtype))
            self.result = Table(
                self.generate_table_name("centers"), dshape=dshape, create=True
            )
            self.result.resize(self.mbk.cluster_centers_.shape[0])
        self.psdict[cols] = self.mbk.cluster_centers_  # type: ignore
        if is_conv:
            return self._return_run_step(self.state_blocked, iter_)
        return self._return_run_step(self.state_ready, iter_)

    def to_json(self, short: bool = False, with_speed: bool = True) -> JSon:
        json = super().to_json(short, with_speed)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json: JSon) -> JSon:
        json["cluster_centers"] = self.table.to_json()
        return json

    def set_centroid(self, c: int, values: List[float]) -> List[float]:
        try:
            c = int(c)
        except ValueError:
            pass

        centroids = self.table
        # idx = centroids.id_to_index(c)

        dfslot = self.get_input_slot("table")
        input_df = dfslot.data()
        columns = self.get_columns(input_df, "table")
        if len(values) != len(columns):
            raise ProgressiveError(f"Expected {len(columns)} values, received {values}")
        centroids.loc[c, columns] = values
        # TODO unpack the table
        centers = centroids.loc[c, columns]
        assert isinstance(centers, BaseTable)
        self.mbk.cluster_centers_[c] = list(centers)
        return self.mbk.cluster_centers_.tolist()

    def create_dependent_modules(self, input_module: Module, input_slot="result"):
        with self.tagged(self.TAG_DEPENDENT):
            s = self.scheduler()
            self.input_module = input_module
            self.input.table = input_module.output[input_slot]
            self.input_slot = input_slot
            c = DynVar(group=self.name, scheduler=s)
            self.moved_center = c
            self.input.moved_center = c.output.result
            v = Var(group=self.name, scheduler=s)
            self.variance = v
            v.input.table = input_module.output[input_slot]
            self.input.var = v.output.result


class MBKMeansFilter(TableModule):
    """
    Filters data corresponding to a specific label
    """

    inputs = [
        SlotDescriptor("table", type=Table, required=True),
        SlotDescriptor("labels", type=Table, required=True),
    ]

    def __init__(self, sel, **kwds):
        self._sel = sel
        super().__init__(**kwds)

    @process_slot("table", "labels")
    @run_if_any
    def run_step(
        self, run_number: int, step_size: int, howlong: float
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
                self.result = TableSelectedView(
                    ctx.table.data(), ctx.labels.data().selection
                )
            else:
                self.selected.selection = ctx.labels.data().selection
            return self._return_run_step(self.next_state(ctx.table), steps_run=steps)

    def create_dependent_modules(self, mbkmeans, data_module, data_slot):
        with self.tagged(self.TAG_DEPENDENT):
            scheduler = self.scheduler()
            filter_ = FilterMod(expr=f"labels=={self._sel}", scheduler=scheduler)
            filter_.input.table = mbkmeans.output.labels
            self.filter = filter_
            self.input.labels = filter_.output.result
            self.input.table = data_module.output[data_slot]
