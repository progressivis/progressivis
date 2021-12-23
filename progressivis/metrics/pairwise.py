# -*- coding: utf-8 -*-
""" Computes the distance matrix from each row of a data frame.
"""
from progressivis.utils.errors import ProgressiveError
from progressivis.core.utils import indices_len, fix_loc
from progressivis.table.module import TableModule, SlotDescriptor
from progressivis.table.table import Table


import pandas as pd
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances  # type: ignore

import logging

logger = logging.getLogger(__name__)


class PairwiseDistances(TableModule):
    inputs = [SlotDescriptor("table", type=Table)]

    def __init__(self, metric="euclidean", columns=None, n_jobs=1, **kwds):
        super(PairwiseDistances, self).__init__(dataframe_slot="distance", **kwds)
        self.default_step_size = kwds.get("step_Size", 100)  # initial guess
        self.columns = columns
        self._metric = metric
        self._n_jobs = n_jobs
        self._last_index = None
        if (
            metric not in _VALID_METRICS
            and not callable(metric)
            and metric != "precomputed"
        ):
            raise ProgressiveError("Unknown distance %s", metric)
        self._table = Table(
            self.generate_table_name("distance"),
            dshape="{distance: var * real}",
            scheduler=self.scheduler(),
            storagegroup=self.storagegroup,
        )

    def is_ready(self):
        if self.get_input_slot("table").created.any():
            return True
        return super(PairwiseDistances, self).is_ready()

    def dist(self):
        return self._table

    def get_data(self, name):
        if name == "dist":
            return self.dist()
        return super(PairwiseDistances, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot("table")
        df = dfslot.data()
        # dfslot.update(run_number)
        if dfslot.updated.any() or dfslot.deleted.any():
            dfslot.reset()
            logger.info("Reseting history because of changes in the input")
            dfslot.update(run_number)
            # TODO: be smarter with changed values

        m = step_size
        indices = dfslot.created.next(m)
        m = indices_len(indices)

        i = None
        j = None
        Si = self._table["document"]

        arrayslot = self.get_input_slot("array")
        if arrayslot is not None and arrayslot.data() is not None:
            array = arrayslot.data()
            logger.debug("Using array instead of DataFrame columns")
            if Si is not None:
                i = array[self._last_index]
            j = array[indices]
        if j is None:
            if self.columns is None:
                self.columns = list(df.columns)
            elif not isinstance(self.columns, pd.Index):
                self.columns = pd.Index(self.columns)
            rows = df[self.columns]
            if Si is not None:
                i = rows.loc[self._last_index]
                assert len(i) == len(self._last_index)
            j = rows.loc[fix_loc(indices)]
            assert len(j) == indices_len(indices)

        Sj = pairwise_distances(j, metric=self._metric, n_jobs=self._n_jobs)
        if Si is None:
            mat = self._buf.resize(Sj.shape[0])
            mat[:, :] = Sj
            self._last_index = dfslot.last_index[indices]
        else:
            Sij = pairwise_distances(i, j, metric=self._metric, n_jobs=self._n_jobs)
            n0 = i.shape[0]
            n1 = n0 + j.shape[0]
            mat = self._buf.resize(n1)
            mat[0:n0, n0:n1] = Sij
            mat[n0:n1, 0:n0] = Sij.T
            mat[n0:n1, n0:n1] = Sj
            self._last_index = self._last_index.append(df.index[indices])
            # truth = pairwise_distances(array[0:n1], metric=self._metric)
            # import pdb
            # pdb.set_trace()
            # assert np.allclose(mat,truth)
        return self._return_run_step(self.next_state(dfslot), steps_run=m)
