from __future__ import print_function
from progressivis import ProgressiveError, SlotDescriptor
from progressivis.core.utils import indices_len, fix_loc
from ..table.module import TableModule
from ..table import Table
from ..table.dshape import dshape_from_dtype

from sklearn.utils import gen_batches
from sklearn.cluster import MiniBatchKMeans

import logging
logger = logging.getLogger(__name__)

import numpy as np

class MBKMeans(TableModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    def __init__(self, n_clusters, columns=None, batch_size=100, tol=0.0, is_input=True, random_state=None,**kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('table', type=Table, required=True)])
        self._add_slots(kwds,'output_descriptors',
                         [SlotDescriptor('labels', type=Table, required=False)])
        super(MBKMeans, self).__init__(**kwds)
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                   verbose=True,
                                   tol=tol, random_state=random_state)
        self.columns = columns
        self.n_clusters = n_clusters
        self.default_step_size = 100
        self._labels = None
        self._remaining_inits = 10
        self._initialization_steps = 0
        self._is_input = is_input

    def reset(self, init='k-means++'):
        print("Reset, init=", init)
        self.mbk = MiniBatchKMeans(n_clusters=self.mbk.n_clusters,
                                   batch_size=self.mbk.batch_size,
                                   init=init,
                                   #tol=self._rel_tol,
                                   random_state=self.mbk.random_state)
        dfslot = self.get_input_slot('table')
        dfslot.reset(mid=self.id)
        self._table = None
        self._labels = None
        self.set_state(self.state_ready)

    def validate_outputs(self):
        valid = super(MBKMeans, self).validate_inputs()
        if valid:
            opt_slot = self.get_output_slot('labels')
            if opt_slot:
                logger.debug('Maintaining labels')
                self.maintain_labels(True)
            else:
                logger.debug('Not maintaining labels')
                self.maintain_labels(False)
        return valid

    def maintain_labels(self, yes=True):
        if yes and self._labels is None:
            self._labels = Table(self.generate_table_name('labels'),
                                dshape="{ labels: int64 }",
#                                scheduler=self.scheduler(),
                                create=True)
        elif not yes:
            self._labels = None

    def labels(self):
        return self._labels

    def get_data(self, name):
        if name=='labels':
            return self.labels()
        return super(MBKMeans, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('table')
        dfslot.update(run_number, self.id)

        if dfslot.deleted.any() or dfslot.updated.any():
            logger.debug('has deleted or updated, reseting')
            self.reset()
            dfslot.update(run_number, self.id)

        #print('dfslot has buffered %d elements'% dfslot.created_length())
        input_df = dfslot.data()
        if (input_df is None or len(input_df)==0) and dfslot.created_length() < self.mbk.n_clusters:
            # Should add more than k items per loop
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.created.next(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        cols = self.get_columns(input_df)
        if len(cols)==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        locs = fix_loc(indices)
        if self._labels is not None and isinstance(indices, slice):
            indices = np.arange(indices.start, indices.stop)
            
        X = input_df.to_array(columns=cols, keys=locs)
        
        batch_size = self.mbk.batch_size or 100
        for batch in gen_batches(steps, batch_size):
            self.mbk.partial_fit(X[batch])
            if self._labels is not None:
                self._labels.append({'labels': self.mbk.labels_},indices=indices[batch])
        if self._table is None:
            dshape = self.dshape_from_columns(input_df, cols, dshape_from_dtype(X.dtype))
            self._table = Table(self.generate_table_name('centers'),
                                dshape=dshape,
                                create=True)
            self._table.resize(self.mbk.cluster_centers_.shape[0])
        self._table[cols] = self.mbk.cluster_centers_
        return self._return_run_step(self.next_state(dfslot), steps_run=steps)

    def dshape_from_columns(self, table, columns, dshape):
        dshapes = []
        for colname in columns:
            col = table._column(colname)
            if len(col.shape) > 1:
                dshapes.append("%s: %d * %s"%(col.name, col.shape[1], dshape))
            else:
                dshapes.append("%s: %s"%(col.name, dshape))
        return "{" + ",".join(dshapes)+"}"

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super(MBKMeans, self).to_json(short)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        if self._table is not None:
            json['cluster_centers'] = self._table.to_json()
        return json

    def set_centroid(self, c, values):
        try:
            c = int(c)
        except:
            pass

        centroids = self._table
        #idx = centroids.id_to_index(c)

        if len(values)!=len(self.columns):
            raise ProgressiveError('Expected %s of values, received %s', len(self.columns), values)
        _ = self.scheduler().for_input(self)
        centroids.loc[c, self.columns] = values
        #TODO unpack the table
        self.mbk.cluster_centers_[c] = centroids.loc[c, self.columns]
        return values

    def is_input(self):
        return self._is_input

    def from_input(self, msg):
        logger.info('Received message %s', msg)
        for c in msg:
            self.set_centroid(c, msg[c])
        #self.reset(init=self.mbk.cluster_centers_)

