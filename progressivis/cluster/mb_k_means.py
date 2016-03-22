from progressivis import ProgressiveError, DataFrameModule, SlotDescriptor
from progressivis.core.buffered_dataframe import BufferedDataFrame
from progressivis.core.utils import indices_len

import numpy as np
import pandas as pd

from sklearn.utils.extmath import squared_norm
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans

import logging
logger = logging.getLogger(__name__)

class MBKMeans(DataFrameModule):
    """
    Mini-batch k-means using the sklearn implementation.
    """
    def __init__(self, n_clusters, columns=None, batch_size=None, tol=None, random_state=None,**kwds):
        self._add_slots(kwds, 'input_descriptors',
                        [SlotDescriptor('df', type=pd.DataFrame, required=True)])
        self._add_slots(kwds,'output_descriptors',
                        [SlotDescriptor('centroids', type=pd.DataFrame, required=False),
                         SlotDescriptor('inertia', type=pd.DataFrame, required=False)])
        super(MBKMeans, self).__init__(**kwds)
        self.mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size,
                                   verbose=True,
                                   tol=tol, random_state=random_state)
        self.columns = columns
        self.n_clusters = n_clusters
        self._df = BufferedDataFrame()
        self._centroids = None
        self._inertia = BufferedDataFrame()
        self._rel_tol = tol
        self.default_step_size = 100

    def df(self):
        return self._df.df()

    def inertia(self):
        return self._inertia.df()

    def centroids(self):
        return self._centroids
        
    def get_data(self, name):
        if name=='centroids':
            return self.centroids()
        if name=='inertia':
            return self.inertia()
        return super(MBKMeans, self).get_data(name)

    def run_step(self, run_number, step_size, howlong):
        dfslot = self.get_input_slot('df')
        dfslot.update(run_number)
        changed = (dfslot.has_created() or dfslot.has_updated())

        if dfslot.has_deleted() or dfslot.has_updated():
            logger.debug('has deleted or updated, reseting')
            dfslot.reset()
            dfslot.update(run_number)
            # need to reset the mini batch as well
            self.mbk = MiniBatchKMeans(n_clusters=self.mbk.n_clusters,
                                       batch_size=self.mbk.batch_size,
                                       tol=self._rel_tol,
                                       random_state=self.mbk.random_state)
            self._buffer.reset()
            self._df = BufferedDataFrame()
            self._inertia = BufferedDataFrame()
            self._centroids = None
            changed = True

        input_df = dfslot.data()
        print 'step_size:', step_size
        indices = dfslot.next_created(step_size) # returns a slice
        steps = indices_len(indices)
        if steps==0:
            return self._return_run_step(self.state_blocked, steps_run=0)
        if isinstance(indices,slice):
            last = indices.stop
            indices=slice(indices.start,indices.stop-1) # semantic of slice with .loc
        else:
            last = indices[-1]
        if self.columns is None:
            self.columns = input_df.columns.difference([self.UPDATE_COLUMN])
        X = input_df[self.columns]
        self.mbk.partial_fit(X.loc[indices].values)
        # bootstrapping
        random_state = check_random_state(self.mbk.random_state)
        for i in range(6):
            new_indices = random_state.random_integers(0, last, self.mbk.batch_size)
            print new_indices
            self.mbk.partial_fit(X.loc[new_indices].values)

        centroids = pd.DataFrame(self.mbk.cluster_centers_.copy(), columns=self.columns)
        if self._centroids is not None:
            centroids_nocol = self._centroids[self.columns]
            centroids[self.UPDATE_COLUMN] = run_number
            #centroids.loc[centroids!=centroids_nocol,self.UPDATE_COLUMN] = run_number
        else:
            centroids[self.UPDATE_COLUMN] = run_number
        self._centroids = centroids
        df = pd.DataFrame({'labels': self.mbk.labels_})
        print 'len(df)=',len(df)
        print 'steps=', steps
        df[self.UPDATE_COLUMN] = run_number
        with self.lock:
            self._df.append(df)
            self._inertia.append(pd.DataFrame({'inertia': [self._inertia],
                                               self.UPDATE_COLUMN: run_number }))
        return self._return_run_step(dfslot.next_state(), steps_run=steps)

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
        return values

    def is_input(self):
        return True

    def from_input(self, msg):
        logger.info('Received message %s', msg)
        for c in msg:
            self.set_centroid(c, msg[c])
