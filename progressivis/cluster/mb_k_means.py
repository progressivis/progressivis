from progressivis import ProgressiveError, DataFrameModule, SlotDescriptor

import numpy as np
import pandas as pd

from sklearn.utils.extmath import squared_norm
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster.k_means_ import _tolerance

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
        self._rel_tol = tol
        self._tol = 0
        self.default_step_size = 100

    def get_data(self, name):
        if name=='centroids':
            return self._centroids
        return super(MBKMeans, self).get_data(name)

    def is_ready(self):
        return super(MBKMeans, self).is_ready() or \
          self._current_tol > self._tol

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        changed = (dfslot.has_created() or dfslot.has_updated())

        if dfslot.has_deleted():
            logger.debug('has deleted, reseting')
            dfslot.reset()
            dfslot.update(run_number)
            self._df = None
            self._centroids = None
            self._current_tol = np.inf
            changed = True
        elif self._current_tol < self._tol and \
           not changed:
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
        if changed:
            self._tol = _tolerance(X, self._rel_tol)
        steps = 0
        if self._df is None:
            logger.info('initializing mini batch k-means')
            # bound time for initialization
            self.mbk.fit(X[0:self.mbk.batch_size])
            steps = 1
        logger.debug('running %s steps', step_size)
        while steps < step_size: # and tol > self._tol:
            self.mbk.partial_fit(X)
            steps += 1

        centroids = pd.DataFrame(self.mbk.cluster_centers_.copy(), columns=self.columns)
        if self._centroids is not None:
            centroids_nocol = self._centroids[self.columns]
            # print 'old centroids:'
            # print centroids_nocol
            # print 'new centroids:'
            # print centroids
            self._current_tol = squared_norm(centroids - centroids_nocol)
            centroids[self.UPDATE_COLUMN] = run_number
            #centroids.loc[centroids!=centroids_nocol,self.UPDATE_COLUMN] = run_number
        else:
            centroids[self.UPDATE_COLUMN] = run_number
        self._centroids = centroids
        df = pd.DataFrame(self.mbk.labels_)
        #TODO optimize if the labels have not changed
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        logger.debug('Tolerance: %s', self._current_tol)
        if self._current_tol < self._tol:
            logger.debug('Tolerance is good: %s < %s', self._current_tol, self._tol)
            return self._return_run_step(self.state_blocked, steps_run=steps)
        logger.debug('Tolerance not good enough: %s > %s', self._current_tol, self._tol)
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

    def set_centroid(self, c, values):
        try:
            c = int(c)
        except:
            pass

        centroids = self._centroids
        if c not in centroids.index:
            raise ProgressiveError('Expected %s values, received %s', len(self.columns), values)

        if len(values)!=len(self.columns):
            raise ProgressiveError('Expected %s of values, received %s', len(self.columns), values)
        run_number = self.scheduler().for_input(self)
        centroids.loc[c, self.columns] = values
        centroids.loc[c, self.UPDATE_COLUMN] = run_number
        self.mbk.cluster_centers_[c] = centroids.loc[c, self.columns]
        #print self.mbk.cluster_centers_
        #print centroids
        self._current_tol = np.inf
        return values

    def is_input(self):
        return True

    def from_input(self, msg):
        logger.info('Received message %s', msg)
        for c in msg:
            self.set_centroid(c, msg[c])
