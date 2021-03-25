import logging

from collections import deque
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster._kmeans import _mini_batch_convergence
from sklearn.utils.validation import _check_sample_weight
from progressivis import ProgressiveError, SlotDescriptor
from progressivis.core.utils import indices_len
from ..table.module import TableModule
from ..table import Table, TableSelectedView
from ..table.dshape import dshape_from_dtype, dshape_from_columns
from ..io import DynVar
from ..utils.psdict import PsDict
from ..core.decorators import process_slot, run_if_any
from ..table.filtermod import FilterMod
from ..stats import Var
logger = logging.getLogger(__name__)

SEED = 42


class MBKMeans(TableModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    parameters = [('samples',  np.dtype(int), 50)]
    inputs = [
        SlotDescriptor('table', type=Table, required=True),
        SlotDescriptor('var', type=Table, required=True),
        SlotDescriptor('moved_center', type=PsDict, required=False)
    ]
    outputs = [
        SlotDescriptor('labels', type=Table, required=False),
        SlotDescriptor('conv', type=PsDict, required=False)
    ]

    DATA_CHANGED_MAX = 4

    def __init__(self, n_clusters, columns=None, batch_size=100, tol=0.01,
                 conv_steps=2,
                 is_input=True, is_greedy=True, random_state=None, **kwds):
        super().__init__(**kwds)
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters,
                                   batch_size=batch_size,
                                   verbose=True,
                                   tol=tol,
                                   random_state=random_state)
        self._random_state = np.random.RandomState(SEED)
        self.columns = columns
        self.n_clusters = n_clusters
        self.default_step_size = 100
        self._labels = None
        self._remaining_inits = 10
        self._initialization_steps = 0
        self._is_input = is_input
        self._tol = tol
        self._conv_steps = conv_steps
        self._old_centers = deque(maxlen=conv_steps)
        self._data_changed = 0
        self._conv_out = PsDict({'convergence': 'unknown'})
        self.params.samples = n_clusters
        self._sample_weight = None
        self._min_p = None
        self._max_p = None
        self._is_greedy = is_greedy

    def predict_step_size(self, duration):
        p = super().predict_step_size(duration)
        return max(p, self.n_clusters)

    def reset(self, init='k-means++'):
        self.mbk = MiniBatchKMeans(n_clusters=self.mbk.n_clusters,
                                   batch_size=self.mbk.batch_size,
                                   init=init,
                                   # tol=self._rel_tol,
                                   random_state=self.mbk.random_state)
        self._random_state = np.random.RandomState(SEED)
        dfslot = self.get_input_slot('table')
        dfslot.reset()
        self._labels = None
        self.set_state(self.state_ready)
        self._data_changed = 0
        self._old_centers.clear()

    def starting(self):
        super().starting()
        opt_slot = self.get_output_slot('labels')
        if opt_slot:
            logger.debug('Maintaining labels')
            self.maintain_labels(True)
        else:
            logger.debug('Not maintaining labels')
            self.maintain_labels(False)

    def maintain_labels(self, yes=True):
        if yes and self._labels is None:
            self._labels = Table(self.generate_table_name('labels'),
                                 dshape="{labels: int64}",
                                 create=True)
        elif not yes:
            self._labels = None

    def labels(self):
        return self._labels

    def get_data(self, name):
        if name == 'labels':
            return self.labels()
        if name == 'conv':
            return self._conv_out
        return super().get_data(name)

    def is_greedy(self):
        return self._is_greedy

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        varslot = self.get_input_slot('var')
        moved_center = self.get_input_slot('moved_center')
        init_centers = 'k-means++'
        if moved_center is not None:
            moved_center.update(run_number)
            if moved_center.has_buffered():
                print("Moved center!!")
                moved_center.clear_buffers()
                msg = moved_center.data()
                for c in msg:
                    self.set_centroid(c, msg[c][:2])
                init_centers = self.mbk.cluster_centers_
                self.reset(init=init_centers)
                dfslot.update(run_number)
        if dfslot.has_buffered() or varslot.has_buffered():
            logger.debug('has deleted or updated, reseting')
            self.reset(init=init_centers)
            dfslot.clear_buffers()
            varslot.clear_buffers()
        # print('dfslot has buffered %d elements'% dfslot.created_length())
        input_df = dfslot.data()
        var_data = varslot.data()
        if (input_df is None or
           var_data is None or
           len(input_df) < self.mbk.n_clusters):
            # Not enough data yet ...
            return self._return_run_step(self.state_blocked, steps_run=0)
        cols = self.get_columns(input_df)
        n_features = len(cols)
        batch_size = self.mbk.batch_size or 100
        n_samples = len(input_df)
        convergence_context = {}
        is_conv = False
        if self._tol > 0:
            v = np.array(list(var_data.last().values()))
            tol = np.mean(v) * self._tol
            old_center_buffer = np.zeros((self.n_clusters, n_features),
                                         dtype=np.float64)
        else:
            tol = 0
            old_center_buffer = np.zeros(0, dtype=np.float64)
        for iter_ in range(step_size):
            mb_ilocs = self._random_state.randint(
                0, n_samples, batch_size)
            mb_locs = input_df.index[mb_ilocs]
            X = input_df.to_array(columns=cols, locs=mb_locs)
            sample_weight = _check_sample_weight(self._sample_weight, X,
                                                 dtype=X.dtype)
            if hasattr(self.mbk, 'cluster_centers_'):
                old_center_buffer[:, :] = self.mbk.cluster_centers_
            self.mbk.partial_fit(X)
            centers = self.mbk.cluster_centers_
            # x_squared_norms = row_norms(X, squared=True)
            nearest_center, batch_inertia = self.mbk.labels_, self.mbk.inertia_
            k = centers.shape[0]
            squared_diff = 0.0
            for center_idx in range(k):
                center_mask = nearest_center == center_idx
                wsum = sample_weight[center_mask].sum()
                if wsum > 0:
                    diff = centers[center_idx].ravel() - \
                      old_center_buffer[center_idx].ravel()
                    squared_diff += np.dot(diff, diff)
            if _mini_batch_convergence(self.mbk,
                                       iter_, step_size, tol, n_samples,
                                       squared_diff, batch_inertia,
                                       convergence_context,
                                       verbose=self.mbk.verbose):
                is_conv = True
                break
        if self.result is None:
            dshape = dshape_from_columns(input_df, cols,
                                         dshape_from_dtype(X.dtype))
            self.result = Table(self.generate_table_name('centers'),
                                dshape=dshape,
                                create=True)
            self.result.resize(self.mbk.cluster_centers_.shape[0])
        self.result[cols] = self.mbk.cluster_centers_
        if is_conv:
            ret_args = (self.state_ready, iter_)
        else:
            ret_args = (self.state_blocked, 0)
        return self._return_run_step(*ret_args)

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super().to_json(short)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        if self.result is not None:
            json['cluster_centers'] = self.result.to_json()
        return json

    def set_centroid(self, c, values):
        try:
            c = int(c)
        except ValueError:
            pass

        centroids = self.result
        # idx = centroids.id_to_index(c)

        if len(values) != len(self.columns):
            raise ProgressiveError('Expected %s of values, received %s',
                                   len(self.columns), values)
        centroids.loc[c, self.columns] = values
        # TODO unpack the table
        self.mbk.cluster_centers_[c] = list(centroids.loc[c, self.columns])
        return self.mbk.cluster_centers_.tolist()

    def create_dependent_modules(self, input_module, input_slot='result'):
        s = self.scheduler()
        self.input_module = input_module
        self.input.table = input_module.output[input_slot]
        self.input_slot = input_slot
        c = DynVar(group="bar", scheduler=s)
        self.moved_center = c
        self.input.moved_center = c.output.result
        v = Var(group="bar", scheduler=s)
        self.variance = v
        v.input.table = input_module.output[input_slot]
        self.input.var = v.output.result


class MBKMeansFilter(TableModule):
    """
    Filters data corresponding to a specific label
    """
    inputs = [
        SlotDescriptor('table', type=Table, required=True),
        SlotDescriptor('labels', type=Table, required=True)
    ]

    def __init__(self, sel, **kwds):
        self._sel = sel
        super().__init__(**kwds)

    @process_slot("table", "labels")
    @run_if_any
    def run_step(self, run_number, step_size, howlong):
        with self.context as ctx:
            indices_t = ctx.table.created.next(step_size)  # returns a slice
            steps_t = indices_len(indices_t)
            ctx.table.clear_buffers()
            indices_l = ctx.labels.created.next(step_size)  # returns a slice
            steps_l = indices_len(indices_l)
            ctx.labels.clear_buffers()
            steps = steps_t + steps_l
            if steps == 0:
                return self._return_run_step(self.state_blocked, steps_run=0)
            if self.result is None:
                self.result = TableSelectedView(ctx.table.data(),
                                                ctx.labels.data().selection)
            else:
                self.result.selection = ctx.labels.data().selection
            return self._return_run_step(self.next_state(ctx.table),
                                         steps_run=steps)

    def create_dependent_modules(self, mbkmeans, data_module, data_slot):
        scheduler = self.scheduler()
        filter_ = FilterMod(expr=f'labels=={self._sel}', scheduler=scheduler)
        filter_.input.table = mbkmeans.output.labels
        self.filter = filter_
        self.input.labels = filter_.output.result
        self.input.table = data_module.output[data_slot]
