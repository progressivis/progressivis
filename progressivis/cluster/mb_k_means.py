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
    schema = [('array', np.dtype(object), None),
              DataFrameModule.UPDATE_COLUMN_DESC]

    def __init__(self, columns, n_clusters, batch_size=100, tol=1e-4, **kwds):
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters,batch_size=batch_size,tol=tol)
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

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)

        if dfslot.has_deleted():
            dfslot.reset()
            dfslot.update(run_number)
            self._df = None
            self._centroids = None
        if self._current_tol < self.mbk.tol and \
           not (dfslot.has_created() or dfslot.has_updated()):
            logger.debug('converged at run_step(%d), tol=%s', run_step, self._current_tol)
            return self._return_run_step(self.state_blocked, steps_run=0)

        input_df = dfslot.data()
        if len(dfslot.data()) < self.mbk.n_clusters:
               return self._return_run_step(self.state_blocked, steps_run=0)
        input_df = input_df[self.columns]
        steps = 0
        if self._df is None:
            # bound time for initialization
            self.mbk.fit(X=input_df.loc[0:self.mbk.batch_size].as_matrix())
            steps = 1
        while steps < step_size:
            self.mbk.partial_fit(X=input_df.as_matrix())
            steps += 1

        centroids = self.get_cluster_centers_df()
        if self._centroids is not None:
            self._current_tol = squared_norm(self._centroids - centroids)
        self._centroids = centroids
        self._centroids[self.UPDATE_COLUMN] = run_number
        df = pd.DataFrame(self.mbk.labels_)
        #TODO optimize if the labels have not changed
        df[self.UPDATE_COLUMN] = run_number
        self._df = df
        return self._return_run_step(dfslot.next_state(), steps_run=steps)

    def get_cluster_centers(self):
        try:
            return self.mbk.cluster_centers_ 
        except AttributeError:
            return pd.Series([])

    def get_cluster_centers_df(self):
        try:
            return pd.DataFrame(self.mbk.cluster_centers_, columns=self.columns)
        except AttributeError:
            return pd.DataFrame([], columns=self.columns)

    def is_visualization(self):
        return False

    def to_json(self, short=False):
        json = super(MBKMeans, self).to_json(short)
        if short:
            return json
        return self._centers_to_json(json)

    def _centers_to_json(self, json):
        json['cluster_centers'] = self.get_cluster_centers_df().to_json()
        return json

