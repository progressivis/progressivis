from progressivis import ProgressiveError, DataFrameModule, SlotDescriptor
from progressivis.core.buffered_dataframe import BufferedDataFrame
from progressivis.core.utils import indices_len

import numpy as np
import pandas as pd

from sklearn.utils.extmath import squared_norm
from sklearn.utils import check_random_state, gen_batches
from sklearn.cluster import MiniBatchKMeans

import logging
logger = logging.getLogger(__name__)

class MBKMeans(DataFrameModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    def __init__(self, n_clusters, columns=None, batch_size=100, tol=0.0, is_input=True, random_state=None,**kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                         [SlotDescriptor('labels', type=pd.DataFrame, required=False)])
        super(MBKMeans, self).__init__(**kwds)
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                   verbose=True,
                                   tol=tol, random_state=random_state)
        self.columns = columns
        self.n_clusters = n_clusters
        self.default_step_size = 100
        self._buffer = None
        self._labels = None
        self._is_input = is_input

    def reset(self, init='k-means++'):
        print "Reset, init=", init
        self.mbk = MiniBatchKMeans(n_clusters=self.mbk.n_clusters,
                                   batch_size=self.mbk.batch_size,
                                   init=init,
                                   #tol=self._rel_tol,
                                   random_state=self.mbk.random_state)
        dfslot = self.get_input_slot('df')
        dfslot.reset()
        if self._buffer is not None:
            self._buffer.reset()
        self._df = None
        self._labels = None

    def validate_outputs(self):
        valid = super(MBKMeans, self).validate_inputs()
        if valid:
            opt_slot = self.get_output_slot('labels')
            if opt_slot:
                logger.debug('Maintaining labels')
                self._buffer = BufferedDataFrame()
            else:
                logger.debug('Not maintaining labels')
        return valid

    def labels(self):
        return self._labels

    def get_data(self, name):
        if name=='labels':
            return self.labels()
        return super(MBKMeans, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)

        if dfslot.has_deleted() or dfslot.has_updated():
            logger.debug('has deleted or updated, reseting')
            self.reset()
            dfslot.update(run_number)

        print('dfslot has buffered %d elements'% dfslot.created_length())
        if dfslot.created_length() < self.mbk.n_clusters:
            # Should add more than k items per loop
            return self._return_run_step(self.state_blocked, steps_run=0)
        indices = dfslot.next_created(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if isinstance(indices,slice):
            indices=slice(indices.start,indices.stop-1) # semantic of slice with .loc

        input_df = dfslot.data()
        if self.columns is None:
            self.columns = input_df.columns.difference([self.UPDATE_COLUMN])
        X = input_df.loc[indices,self.columns].values
        batch_size = self.mbk.batch_size or 100
        for batch in gen_batches(steps, batch_size):
            self.mbk.partial_fit(X[batch])
            if self._buffer is not None:
                df = pd.DataFrame({'labels': self.mbk.labels_})
                df[self.UPDATE_COLUMN] = run_number
                self._buffer.append(df)

        with self.lock:
            self._df = pd.DataFrame(self.mbk.cluster_centers_, columns=self.columns)
            self._df[self.UPDATE_COLUMN] = run_number
            if self._buffer is not None:
                logger.debug('Setting the labels')
                self._labels = self._buffer.df()
        return self._return_run_step(dfslot.next_state(), steps_run=steps)

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super(MBKMeans, self).to_json(short)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        if self._df is not None:
            json['cluster_centers'] = self._df.to_json()
        return json

    def set_centroid(self, c, values):
        try:
            c = int(c)
        except:
            pass

        centroids = self._df
        if c not in centroids.index:
            raise ProgressiveError('Expected %s values, received %s', len(self.columns), values)

        if len(values)!=len(self.columns):
            raise ProgressiveError('Expected %s of values, received %s', len(self.columns), values)
        run_number = self.scheduler().for_input(self)
        centroids.loc[c, self.columns] = values
        centroids.loc[c, self.UPDATE_COLUMN] = run_number
        self.mbk.cluster_centers_[c] = centroids.loc[c, self.columns]
        return values

    def is_input(self):
        return self._is_input

    def from_input(self, msg):
        logger.info('Received message %s', msg)
        for c in msg:
            self.set_centroid(c, msg[c])
        self.reset(init=self.mbk.cluster_centers_)
