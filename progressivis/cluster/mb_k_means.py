from progressivis import DataFrameModule, SlotDescriptor

import numpy as np
import pandas as pd

from sklearn.utils.extmath import squared_norm
from sklearn.cluster import MiniBatchKMeans

import logging
logger = logging.getLogger(__name__)

class MBKMeans(DataFrameModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    def __init__(self, n_clusters, columns=None, batch_size=100, tol=1e-4, random_state=None,**kwds):
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                   tol=tol, random_state=random_state)
        self._add_slots(kwds, 'input_descriptors',
                [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('centroids', type=pd.DataFrame, required=False)])
        super(MBKMeans, self).__init__(**kwds)
        self.columns = columns
        self.n_clusters = n_clusters
        self._df = None
        self._centroids = None
        self._current_tol = np.inf

    def get_data(self, name):
        if name=='centroids':
            return self._centroids
        return super(MBKMeans, self).get_data(name)

    def is_ready(self):
        if self._current_tol > self.mbk.tol:
            return True
        return super(MBKMeans, self).is_ready()

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)

        if dfslot.has_deleted():
            dfslot.reset()
            dfslot.update(run_number)
            self._df = None
            self._centroids = None
            self._current_tol = np.inf
        elif self._current_tol < self.mbk.tol and \
           not (dfslot.has_created() or dfslot.has_updated()):
            logger.debug('converged at run_number(%d), tol=%s', run_number, self._current_tol)
            return self._return_run_step(self.state_blocked, steps_run=0)

        input_df = dfslot.data()
        dfslot.next_created() # just flush the created
        dfslot.next_updated() # and the updated
        if len(dfslot.data()) < self.mbk.n_clusters:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if self.columns is None:
            self.columns = input_df.columns.difference([self.UPDATE_COLUMN])
        X = input_df[self.columns].values
        steps = 0
        if self._df is None:
            # bound time for initialization
            self.mbk.fit(X[0:self.mbk.batch_size])
            steps = 1
        while steps < step_size: # and tol > self.mbk.tol:
            self.mbk.partial_fit(X)
            steps += 1

        centroids = pd.DataFrame(self.mbk.cluster_centers_, columns=self.columns)
        if self._centroids is not None:
            self._current_tol = squared_norm(centroids - self._centroids[self.columns])
        self._centroids = centroids
        self._centroids[self.UPDATE_COLUMN] = run_number
        df = pd.DataFrame(self.mbk.labels_)
        #TODO optimize if the labels have not changed
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        logger.debug('Tolerance: %s', self._current_tol)
        if self._current_tol < self.mbk.tol:
            logger.debug('Tolerance is good')
            if not (dfslot.has_created() or dfslot.has_updated()):
                self._return_run_step(self.state_blocked, steps_run=steps)
            logger.debug('But more work to do, created: %d, updated: %d', dfslot.created_length(), dfslot.updated_length())
        return self._return_run_step(self.state_ready, steps_run=steps)

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super(MBKMeans, self).to_json(short)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        if self._centroids is not None:
            json['cluster_centers'] = self._centroids.to_json()
        return json

