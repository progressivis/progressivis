import logging

from collections import deque
import numpy as np
from sklearn.utils import gen_batches
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial import distance
from progressivis import ProgressiveError, SlotDescriptor
from progressivis.core.utils import indices_len, fix_loc
from ..table.module import TableModule
from ..table import Table, TableSelectedView
from ..table.dshape import dshape_from_dtype, dshape_from_columns
from ..io import DynVar
from ..utils.psdict import PsDict
from ..core.decorators import *
from ..table.filtermod import FilterMod

logger = logging.getLogger(__name__)


class MBKMeans(TableModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    parameters = [('samples',  np.dtype(int), 50)]
    inputs = [
        SlotDescriptor('table', type=Table, required=True),
        SlotDescriptor('moved_center', type=PsDict, required=False)        
    ]
    outputs = [
        SlotDescriptor('labels', type=Table, required=False),
        SlotDescriptor('conv', type=PsDict, required=False)
    ]

    DATA_CHANGED_MAX = 4
    def __init__(self, n_clusters, columns=None, batch_size=100, tol=0.01, conv_steps=2,
                 is_input=True, is_greedy=True, random_state=None, **kwds):
        super().__init__(**kwds)
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters,
                                   batch_size=batch_size,
                                   verbose=True,
                                   tol=tol,
                                   random_state=random_state)
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
        self._min_p = None
        self._max_p = None
        self._is_greedy = is_greedy
        
    def predict_step_size(self, duration):
        p = super().predict_step_size(duration)
        return max(p, self.n_clusters)

    def reset(self, init='k-means++'):
        print("Reset, init=", init)
        self.mbk = MiniBatchKMeans(n_clusters=self.mbk.n_clusters,
                                   batch_size=self.mbk.batch_size,
                                   init=init,
                                   # tol=self._rel_tol,
                                   random_state=self.mbk.random_state)
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

    def _test_convergence(self):
        ampl = distance.euclidean(self._min_p, self._max_p)
        eps_ = ampl*self._tol
        last_centers = self._old_centers[-1]
        def _loop_fun(): # avoids a double break on False case
            for old_centers in list(self._old_centers)[:-1]:
                sum = 0.0
                for i in range(self.mbk.n_clusters):
                    sum += distance.euclidean(old_centers[i], last_centers[i])
                    if sum > eps_:
                        print("Convergence test failed:", sum, self._tol, old_centers[i], last_centers[i], i)
                        return False
            return True
        res = _loop_fun()
        if res: print("Convergence test succeeded")
        self._conv_out['convergence'] = 'yes' if res else 'no'

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        moved_center = self.get_input_slot('moved_center')
        init_centers = 'k-means++'
        if moved_center is not None:
            moved_center.update(run_number)        
            if moved_center.has_buffered():
                print("Moved center!!")
                moved_center.created.next()
                moved_center.updated.next()
                moved_center.deleted.next()            
                msg = moved_center.data()
                for c in msg:
                    self.set_centroid(c, msg[c][:2])
                init_centers = self.mbk.cluster_centers_
                self.reset(init=init_centers)
                dfslot.update(run_number)
        # dfslot.update(run_number)

        if dfslot.deleted.any() or dfslot.updated.any():
            logger.debug('has deleted or updated, reseting')
            self.reset(init=init_centers)
            dfslot.update(run_number)

        # print('dfslot has buffered %d elements'% dfslot.created_length())
        input_df = dfslot.data()
        if (input_df is None or len(input_df)  < self.mbk.n_clusters): #and \
           #dfslot.created_length() < self.mbk.n_clusters:
            # Should add more than k items per loop
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.created.next(step_size)  # returns a slice
        steps = indices_len(indices)
        if steps == 0:
            self._data_changed -= 1
            #print("ZERO STEPS", self._data_changed)
            trm = dfslot.output_module.is_terminated() or dfslot.output_module.is_zombie()
            if self._data_changed==1 or trm:
                print("DATA CHANGED", self._data_changed, trm)
                self._test_convergence()
                args = (self.state_blocked,0) if trm  else (self.state_ready, 1)
                return self._return_run_step(*args)
            return self._return_run_step(self.state_blocked, steps_run=0)
        elif steps >= self.n_clusters:
            #print("DATA_CHANGED_MAX")
            self._data_changed = self.DATA_CHANGED_MAX
        else: # n_samples should be larger than k
            print("n_samples should be larger than k", steps, self.n_clusters, step_size)
            return self._return_run_step(self.state_blocked, steps_run=0)
        cols = self.get_columns(input_df)
        if len(cols) == 0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        locs = fix_loc(indices)
        if self._labels is not None and isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop)

        X = input_df.to_array(columns=cols, locs=locs)
        _min = np.min(X, axis=0)
        _max = np.max(X, axis=0)
        if self._min_p is None:
            self._min_p = _min
            self._max_p = _max
        else:
            self._min_p = np.minimum(self._min_p, _min)
            self._max_p = np.maximum(self._max_p, _max)
            
        batch_size = self.mbk.batch_size or 100
        # print("STEPS...", steps, batch_size)        
        for batch in gen_batches(steps, batch_size):
            self.mbk.partial_fit(X[batch])
            self._old_centers.append(self.mbk.cluster_centers_.copy())
            if self._labels is not None:
                self._labels.append({'labels': self.mbk.labels_},
                                    indices=indices[batch])
        if self.result is None:
            dshape = dshape_from_columns(input_df, cols,
                                              dshape_from_dtype(X.dtype))
            self.result = Table(self.generate_table_name('centers'),
                                dshape=dshape,
                                create=True)
            self.result.resize(self.mbk.cluster_centers_.shape[0])
        self.result[cols] = self.mbk.cluster_centers_
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)

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

    def create_dependent_modules(self):
        c = DynVar(group="bar", scheduler=self.scheduler())
        self.moved_center = c
        self.input.moved_center = c.output.result
        

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
        #import pdb;pdb.set_trace()
        with self.context as ctx:
            indices_t = ctx.table.created.next(step_size) # returns a slice
            steps_t = indices_len(indices_t)
            ctx.table.clear_buffers()
            indices_l = ctx.labels.created.next(step_size) # returns a slice
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
